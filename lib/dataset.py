import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from glob import glob
import numpy as np
import random
import math
import os
import sys
import csv
import cv2
import config

data_dir = config.data_dir
IMG_SIZE = 224
batch_size = 16
_STRIDE = 8
model_name = config.model_name
channel_name = config.channel_name
val_name = config.val_name

def decode_img(img, channel):
  img = tf.io.decode_jpeg(img, channels = channel)
  img = tf.cast(img, tf.float32)
  return tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

def resample(keep_list):
  if len(keep_list) in range(int(_STRIDE / 2) + 1, _STRIDE):
    sample_list = keep_list + random.sample(keep_list, _STRIDE - len(keep_list))
    sample_list.sort()
  else:  
    keep_list = keep_list * (math.ceil(_STRIDE/len(keep_list)) + 1)
    sample_list = [keep_list[0]] + random.sample(keep_list, _STRIDE - 2) + [keep_list[-1]]
    sample_list.sort()
  return ' '.join(['{:05d}.png'.format(x) for x in sample_list])

def get_images(file_path):
  img_list = []
  crop_list = []
  diff_list = []
  for i in range(_STRIDE):
    img_path = tf.strings.join([data_dir, 'Images', tf.strings.split(file_path, sep = ' ')[0], tf.strings.split(file_path, sep = ' ')[i + 1]], separator = '/')
    crop_path = tf.strings.join([data_dir, 'Crop_images', tf.strings.split(file_path, sep = ' ')[0], tf.strings.split(file_path, sep = ' ')[i + 1]], separator = '/')
    diff_path = tf.strings.join([data_dir, 'Diff_images', tf.strings.split(file_path, sep = ' ')[0], tf.strings.split(file_path, sep = ' ')[i + 1]], separator = '/')
    img = tf.io.read_file(img_path)
    img_crop = tf.io.read_file(crop_path)
    img_diff = tf.io.read_file(diff_path)
    img = decode_img(img, 3)
    img_crop = decode_img(img_crop, 3)
    img_diff = decode_img(img_diff, 1)
    img_list.append(img)
    crop_list.append(img_crop)
    diff_list.append(img_diff)
  image_data = tf.concat([tf.stack(img_list), tf.stack(crop_list), tf.stack(diff_list)], axis = 3)
  return image_data

def process_label(file_path):
  image_data = get_images(file_path)
  if model_name == 'all':
    verb_label = tf.strings.to_number(tf.strings.split(file_path, sep = ' ')[9], tf.int32)
    noun_label = tf.strings.to_number(tf.strings.split(file_path, sep = ' ')[10], tf.int32)
    adv_label = tf.map_fn(fn = lambda x: tf.strings.to_number(x, tf.int32), elems = tf.strings.split(file_path, sep = ' ')[11:], fn_output_signature = tf.int32)
    return {'image': image_data, 'adv': adv_label}, {'verb': verb_label, 'noun': noun_label, 'adv': adv_label}
  elif model_name == 'verb_only':
    verb_label = tf.strings.to_number(tf.strings.split(file_path, sep = ' ')[9], tf.int32)
    return {'image': image_data}, {'verb': verb_label}
  elif model_name == 'noun_only':
    noun_label = tf.strings.to_number(tf.strings.split(file_path, sep = ' ')[10], tf.int32)
    return {'image': image_data}, {'noun': noun_label}
  elif model_name == 'adv_only':
    adv_label = tf.map_fn(fn = lambda x: tf.strings.to_number(x, tf.int32), elems = tf.strings.split(file_path, sep = ' ')[11:], fn_output_signature = tf.int32)
    return {'image': image_data, 'adv': adv_label}, {'adv': adv_label}
  elif model_name == 'lstm':
    adv_label = tf.map_fn(fn = lambda x: tf.strings.to_number(x, tf.int32), elems = tf.strings.split(file_path, sep = ' ')[9:], fn_output_signature = tf.int32)
    return {'image': image_data, 'adv': adv_label}, {'adv': adv_label}
  else:
    raise NameError(model_name)

def configure_for_performance(ds, name):
  ds = ds.cache('data/cache/' + model_name + '_' + val_name + '/' + name)
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds

def parse_label(verb_list, noun_list, vocab_list, row):
  if model_name == 'lstm':
    verb_class = str(vocab_list.index(row[2].replace(' ', '').lower()))
    noun_class = str(vocab_list.index(row[3].replace(' ', '').lower()))
  else:
    verb_class = str(verb_list.index(row[2].replace(' ', '').lower()))
    noun_class = str(noun_list.index(row[3].replace(' ', '').lower()))
  adv = row[5].lower() + ' <end>'
  adv_list = [0] * 5
  for i in range(len(adv.split())):
    adv_list[i] = vocab_list.index(adv.split()[i])
  return verb_class, noun_class, ' '.join([str(x) for x in adv_list])

def get_label_files(val_name):
  #train_files = glob(os.path.join(data_dir, 'Labels', 'captions', '[0-9]*'))
  #train_files.extend(glob(os.path.join(data_dir, 'Labels', 'captions', 'windows*')))
  #train_files.extend(glob(os.path.join(data_dir, 'Labels', 'captions', 'photoshop*')))
  train_files = glob(os.path.join(data_dir, 'Labels', 'captions', '*'))
  if val_name == 'all':
    val_files = glob(os.path.join(data_dir, 'Labels', 'captions', '*'))
  elif val_name in ['[0-9]', 'word', 'windows', 'photoshop', 'zoom']:
    val_files = glob(os.path.join(data_dir, 'Labels', 'captions', val_name + '*'))
    for val in val_files:
      if val in train_files:
        train_files.remove(val)
  elif val_name in ['[0-9]_t', 'word_t', 'windows_t', 'photoshop_t', 'zoom_t']:
    val_files = glob(os.path.join(data_dir, 'Labels', 'captions', val_name.split('_')[0] + '*'))
    train_files = val_files[:]
  else:
    raise NameError(val_name)
  return train_files, val_files

def get_label_rows(file_list):
  row_list = []
  for csv_file in tqdm(file_list):
    with open(csv_file, 'r') as csvfile:
      spamreader = csv.reader(csvfile)
      next(spamreader, None)
      for row in spamreader:
        row.append(csv_file.split('.')[0].split('/')[-1])
        row_list.append(row)
  random.shuffle(row_list)
  return row_list

def create_dataset():
  train_files, val_files = get_label_files(val_name)
  label_list_train = []
  label_list_eval = []
  with open(os.path.join(data_dir, 'Labels', 'verbs.txt'), 'r') as verb_file:
    verb_list = [x.strip() for x in verb_file.readlines()]
  with open(os.path.join(data_dir, 'Labels', 'nouns.txt'), 'r') as noun_file:
    noun_list = [x.strip() for x in noun_file.readlines()]
  with open(os.path.join(data_dir, 'Labels', 'vocab.txt'), 'r') as vocab_file:
    vocab_list = [x.strip() for x in vocab_file.readlines()]
  train_rows = get_label_rows(train_files)
  val_rows = get_label_rows(val_files)
  if val_name == 'all':
    val_rows = train_rows[int(len(train_rows) * 0.8):]
    train_rows = train_rows[:int(len(train_rows) * 0.8)]
  if val_name in ['[0-9]_t', 'word_t', 'windows_t', 'photoshop_t', 'zoom_t']:
    val_rows = train_rows[int(len(train_rows) * 0.8):]
    train_rows = train_rows[:int(len(train_rows) * 0.8)]
  for row in train_rows:
    csv_file = row[-1]
    stride = resample(list(np.arange(int(row[0]), int(row[1]) + 1)))
    verb_class, noun_class, adv_class = parse_label(verb_list, noun_list, vocab_list, row)
    label_list_train.append(csv_file + ' ' + stride + ' ' + verb_class + ' ' + noun_class + ' ' + adv_class + '\n')
  for row in val_rows:
    csv_file = row[-1]
    stride = resample(list(np.arange(int(row[0]), int(row[1]) + 1)))
    stride_2 = resample(list(np.arange(int(row[0]), int(row[1]) + 1)))
    verb_class, noun_class, adv_class = parse_label(verb_list, noun_list, vocab_list, row)
    label_list_eval.append(csv_file + ' ' + stride + ' ' + verb_class + ' ' + noun_class + ' ' + adv_class + '\n')

  with open(os.path.join(data_dir, 'Train', model_name + '_' + val_name, 'train_' + '.txt'), 'w') as train_list_file:
    train_list_file.writelines(label_list_train)
  with open(os.path.join(data_dir, 'Train', model_name + '_' + val_name, 'eval_' + '.txt'), 'w') as eval_list_file:
    eval_list_file.writelines(label_list_eval)

def get_dataset():
  if not os.path.exists(os.path.join(data_dir, 'Train', model_name + '_' + val_name, 'train_' + '.txt')):
    if not os.path.exists(os.path.join(data_dir, 'Train', model_name + '_' + val_name)):
      os.makedirs(os.path.join(data_dir, 'Train', model_name + '_' + val_name))
    print ('create dataset ' + model_name + '_' + val_name)
    create_dataset()
  if not os.path.exists(os.path.join(data_dir, 'cache', model_name + '_' + val_name)):
    os.makedirs(os.path.join(data_dir, 'cache', model_name + '_' + val_name))
  with open(os.path.join(data_dir, 'Train', model_name + '_' + val_name, 'train_' + '.txt'), 'r') as train_file:
    train_list = train_file.readlines()
  train_list = [x.strip() for x in train_list]
  with open(os.path.join(data_dir, 'Train', model_name + '_' + val_name, 'eval_' +  '.txt'), 'r') as eval_file:
    eval_list = eval_file.readlines()
  eval_list = [x.strip() for x in eval_list]
  train_ds = tf.data.Dataset.from_tensor_slices(train_list)
  val_ds = tf.data.Dataset.from_tensor_slices(eval_list)
  train_ds = train_ds.map(process_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  val_ds = val_ds.map(process_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_ds = configure_for_performance(train_ds, 'train')
  val_ds = configure_for_performance(val_ds, 'val')
  #image_batch, label_batch = next(iter(train_ds))
  #A, B, C = tf.split(image_batch, num_or_size_splits = 3, axis = 4)
  #print (label_batch)
  #exit(1)
  return train_ds, val_ds

if __name__ == '__main__':
  get_dataset()

