import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Conv3D, MaxPooling3D, MaxPooling2D, Dropout, Softmax, Reshape
from tensorflow.keras import Model
from tensorflow.keras import layers
import config

units = 128
embedding_dim = 256
vocab_size = 85
verb_class = 11
noun_class = 11
channel_name = config.channel_name

class Encoder_3D(Model):
  def __init__(self, layer_name):
    super(Encoder_3D, self).__init__()
    self.rescale = layers.experimental.preprocessing.Rescaling(1./255)
    self.conv1 = Conv3D(16, 3, padding = 'same', activation='relu', input_shape = (8, 224, 224, 3), name = layer_name + 'conv1')
    self.conv2 = Conv3D(32, 3, padding = 'same', activation='relu', name = layer_name + 'conv2')
    self.conv3 = Conv3D(64, 3, padding = 'same', activation='relu', name = layer_name + 'conv3')
    self.conv4 = Conv3D(128, 3, padding = 'same', activation='relu', name = layer_name + 'conv4')
    self.maxpooling1 = MaxPooling3D(pool_size = (1, 2, 2), strides = (1, 2, 2), padding = 'valid', name = layer_name + 'pool1')
    self.maxpooling2 = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding = 'valid', name = layer_name + 'pool2')
    self.conv5 = Conv2D(256, 1, padding = 'same', activation='relu', name = layer_name + 'conv5')
    self.dropout = Dropout(0.5, name = layer_name + 'dropout')
    self.reshape = Reshape((-1, 256))

  def call(self, x):
    x = self.rescale(x)
    x = self.conv1(x)
    x = self.maxpooling1(x)
    x = self.conv2(x)
    x = self.maxpooling2(x)
    x = self.conv3(x)
    x = self.maxpooling2(x)
    x = self.conv4(x)
    x = self.maxpooling2(x)
    x = tf.squeeze(x, 1)
    x = self.conv5(x)
    x = self.reshape(x)
    x = self.dropout(x)
    return x

class Encoder_1D(Model):
  def __init__(self, layer_name):
    super(Encoder_1D, self).__init__()
    self.rescale = layers.experimental.preprocessing.Rescaling(1./255)
    self.conv1 = Conv3D(16, 3, padding = 'same', activation='relu', input_shape = (8, 224, 224, 1), name = layer_name + 'conv1')
    self.conv2 = Conv3D(32, 3, padding = 'same', activation='relu', name = layer_name + 'conv2')
    self.conv3 = Conv3D(64, 3, padding = 'same', activation='relu', name = layer_name + 'conv3')
    self.conv4 = Conv3D(128, 3, padding = 'same', activation='relu', name = layer_name + 'conv4')
    self.maxpooling1 = MaxPooling3D(pool_size = (1, 2, 2), strides = (1, 2, 2), padding = 'valid', name = layer_name + 'pool1')
    self.maxpooling2 = MaxPooling3D(pool_size = (2, 2, 2), strides = (2, 2, 2), padding = 'valid', name = layer_name + 'pool2')
    self.conv5 = Conv2D(256, 1, padding = 'same', activation='relu', name = layer_name + 'conv5')
    self.dropout = Dropout(0.5, name = layer_name + 'dropout')
    self.reshape = Reshape((-1, 256))

  def call(self, x):
    x = self.rescale(x)
    x = self.conv1(x)
    x = self.maxpooling1(x)
    x = self.conv2(x)
    x = self.maxpooling2(x)
    x = self.conv3(x)
    x = self.maxpooling2(x)
    x = self.conv4(x)
    x = self.maxpooling2(x)
    x = tf.squeeze(x, 1)
    x = self.conv5(x)
    x = self.reshape(x)
    x = self.dropout(x)
    return x

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
    score = self.V(attention_hidden_layer)
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

class RNN_Decoder(Model):
  def __init__(self, embedding_dim, units, vocab_size, **kwargs):
    super(RNN_Decoder, self).__init__(**kwargs)
    self.units = units
    self.fc1 = Dense(embedding_dim, activation='relu')
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True)
    self.fc2 = Dense(self.units)
    self.fc3 = Dense(vocab_size)
    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    features = self.fc1(features)
    context_vector, attention_weights = self.attention(features, hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    x = self.fc2(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc3(x)
    return x, state
  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

class VCModel(Model):
  def __init__(self, is_training = True):
    super(VCModel, self).__init__()
    self.is_training = is_training
    self.model_name = config.model_name
    self.image_encoder = Encoder_3D(layer_name = 'image_')
    self.crop_encoder = Encoder_3D(layer_name = 'crop_')
    self.diff_encoder = Encoder_1D(layer_name = 'diff_')
    self.fc_verb = Dense(verb_class, name = 'verb_output')
    self.fc_noun = Dense(noun_class, name = 'noun_output')
    self.flatten = Flatten()
    self.fc1 = Dense(512, activation='relu')
    self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)

  def generate_adv(self, features, target, target_len = 4):
    hidden = self.decoder.reset_state(batch_size = tf.shape(target)[0])
    dec_input = tf.ones([tf.shape(target)[0], 1])
    #predictions, hidden = self.decoder(dec_input, features, hidden)
    #dec_input = verb
    #predictions, hidden = self.decoder(dec_input, features, hidden)
    #dec_input = noun
    predictions, hidden = self.decoder(dec_input, features, hidden)
    output = tf.expand_dims(predictions, 1)
    for i in range(target_len):
      dec_input = tf.expand_dims(target[:, i], 1)
      predictions, hidden = self.decoder(dec_input, features, hidden)
      output = tf.concat([output, tf.expand_dims(predictions, 1)], axis = 1)
    return output

  def generate_adv_evaluate(self, features, target_len = 4):
    hidden = self.decoder.reset_state(batch_size = tf.shape(features)[0])
    dec_input = tf.ones([tf.shape(features)[0], 1])
    predictions, hidden = self.decoder(dec_input, features, hidden)
    #dec_input = tf.expand_dims(tf.math.argmax(verb, 1), 1)
    #predictions, hidden = self.decoder(dec_input, features, hidden)
    #dec_input = tf.expand_dims(tf.math.argmax(noun, 1), 1)
    #predictions, hidden = self.decoder(dec_input, features, hidden)
    predicted_id = tf.random.categorical(predictions, 1)
    output = tf.expand_dims(predictions, 1)
    for i in range(target_len):
      dec_input = predicted_id
      predictions, hidden = self.decoder(dec_input, features, hidden)
      predicted_id = tf.random.categorical(predictions, 1)
      output = tf.concat([output, tf.expand_dims(predictions, 1)], axis = 1)
    return output 

  def extract_feature(self, x):
    image, crop, diff = tf.split(x['image'], num_or_size_splits = [3, 3, 1], axis = 4)
    if channel_name == 'all':
      image_feature = self.image_encoder(image)
      crop_feature = self.crop_encoder(crop)
      diff_feature = self.diff_encoder(diff)
      features = tf.concat([image_feature, crop_feature, diff_feature], axis = 1)
    elif channel_name == 'cropimage':
      image_feature = self.image_encoder(image)
      crop_feature = self.crop_encoder(crop)
      features = tf.concat([image_feature, crop_feature], axis = 1)
    elif channel_name == 'cropdiff':
      crop_feature = self.crop_encoder(crop)
      diff_feature = self.diff_encoder(diff)
      features = tf.concat([crop_feature, diff_feature], axis = 1)
    elif channel_name == 'imagediff':
      image_feature = self.image_encoder(image)
      diff_feature = self.diff_encoder(diff)
      features = tf.concat([image_feature, diff_feature], axis = 1)
    elif channel_name == 'image':
      image_feature = self.image_encoder(image)
      features = image_feature
    elif channel_name == 'crop':
      crop_feature = self.crop_encoder(crop)
      features = crop_feature
    elif channel_name == 'diff':
      diff_feature = self.diff_encoder(diff)
      features = diff_feature
    else:
      raise NameError(channel_name)  
    return features

  def call(self, x):
    features = self.extract_feature(x)
    if self.model_name == 'all':
      flatten_features = self.flatten(features)
      flatten_features = self.fc1(flatten_features)
      noun = self.fc_noun(flatten_features)
      verb = self.fc_verb(flatten_features)
      if self.is_training:
        adv = self.generate_adv(features, x['adv'], target_len = 4)
      else:
        verb = tf.nn.softmax(verb)
        noun = tf.nn.softmax(noun)
        adv = self.generate_adv_evaluate(features, target_len = 4)
      #adv = self.generate_adv(features, x['adv'], target_len = 4)
      return {'verb': verb, 'noun': noun, 'adv': adv}
    elif self.model_name == 'verb_only':
      flatten_features = self.flatten(features)
      flatten_features = self.fc1(flatten_features)
      verb = self.fc_verb(flatten_features)
      if not self.is_training:
        verb = tf.nn.softmax(verb)
      return {'verb': verb}
    elif self.model_name == 'noun_only':
      flatten_features = self.flatten(features)
      flatten_features = self.fc1(flatten_features)
      noun = self.fc_noun(flatten_features)
      if not self.is_training:
        noun = tf.nn.softmax(noun)
      return {'noun': noun}
    elif self.model_name == 'adv_only':
      if self.is_training:
        adv = self.generate_adv(features, x['adv'], target_len = 4)
      else:
        #adv = self.generate_adv(features, x['adv'], target_len = 4)
        adv = self.generate_adv_evaluate(features, target_len = 4)
      return {'adv': adv}
    elif self.model_name == 'lstm':
      if self.is_training:
        adv = self.generate_adv(features, x['adv'], target_len = 6)
      else:
        adv = self.generate_adv_evaluate(features, target_len = 6)
      return {'adv': adv}

