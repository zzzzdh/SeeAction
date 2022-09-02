import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import numpy as np
import itertools
import io
import os
from lib import dataset
from lib.net import VCModel
import config

'''
model_name = 'verb_only'
model_name = 'noun_only'
model_name = 'adv_only'
model_name = 'lstm'
model_name = 'all'
'''

epochs = config.epochs
model_name = config.model_name
val_name = config.val_name
channel_name = config.channel_name
if val_name == '[0-9]':
  val_name = 'firefox'
if val_name == '[0-9]_t':
  val_name = 'firefox_t'

def main():
  train_dataset, val_dataset = dataset.get_dataset()
  model = VCModel(is_training = True)
  monitor = 'val_accuracy'
  if model_name == 'all':
    model.compile(optimizer='adam',
      loss={'verb': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'noun': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'adv': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
            loss_weights=[1, 1, 1],
            metrics=['accuracy'])
    monitor = 'val_loss'
  elif model_name == 'verb_only':
    model.compile(optimizer='adam',
      loss={'verb': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
            metrics=['accuracy'])
  elif model_name == 'noun_only':
    model.compile(optimizer='adam',
      loss={'noun': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
            metrics=['accuracy'])
  elif model_name == 'adv_only' or model_name == 'lstm':
    model.compile(optimizer='adam',
      loss={'adv': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
            metrics=['accuracy'],)
  else:
    raise NameError(model_name)

  callbacks = [
    keras.callbacks.ModelCheckpoint(
      filepath='data/model/' + model_name + '_' + channel_name + '_' + val_name + '/VCModel',
      save_weights_only=True,
      save_best_only=True,
      monitor=monitor,
      mode='auto',
      verbose=1,),
      keras.callbacks.TensorBoard(
        log_dir='logs/' + model_name + '_' + channel_name + '_' + val_name),
      ]

  if os.path.exists(os.path.join('data', 'model', model_name + '_' + channel_name + '_' + val_name)):
    model.load_weights('data/model/' + model_name + '_' + channel_name + '_' + val_name + '/VCModel')

  history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks
  )
  print (history.history)



if __name__ == '__main__':
  main()
