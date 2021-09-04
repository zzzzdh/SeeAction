import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from glob import glob
import numpy as np
import random
import math
import os
import csv
import cv2
import config

data_dir = config.data_dir
IMG_SIZE = 224
batch_size = 16
_STRIDE = 8

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
    img_path = tf.strings.join([data_dir, 'Images', tf.strings.split(file_path, sep = ' ')[1], tf.strings.split(file_path, sep = ' ')[i + 2]], separator = '/')
    crop_path = tf.strings.join([data_dir, 'Crop_images', tf.strings.split(file_path, sep = ' ')[1], tf.strings.split(file_path, sep = ' ')[i + 2]], separator = '/')
    diff_path = tf.strings.join([data_dir, 'Diff_images', tf.strings.split(file_path, sep = ' ')[1], tf.strings.split(file_path, sep = ' ')[i + 2]], separator = '/')
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
  return {'image': image_data, 'name': tf.strings.split(file_path, sep = ' ')[1], 'id': tf.strings.split(file_path, sep = ' ')[0]}
 
def configure_for_performance(ds):
  ds = ds.cache('data/cache/test/test')
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return ds

def get_label_rows(file_list):
  row_list = []
  for csv_file in tqdm(file_list):
    i = 0
    with open(csv_file, 'r') as csvfile:
      spamreader = csv.reader(csvfile)
      next(spamreader, None)
      for row in spamreader:
        row.append(csv_file.split('.')[0].split('/')[-1])
        row.append(str(i))
        row_list.append(row)
        i += 1
  return row_list

def create_dataset():
  test_files = glob(os.path.join(data_dir, 'results', 'segments', '*'))
  label_list_test = []
  test_rows = get_label_rows(test_files)
  for row in test_rows:
    csv_file = row[-2]
    index = row[-1]
    stride = resample(list(np.arange(int(row[0]), int(row[1]) + 1)))
    label_list_test.append(index + ' ' + csv_file + ' ' + stride + '\n')
  with open(os.path.join(data_dir, 'Train', 'test.txt'), 'w') as test_list_file:
    test_list_file.writelines(label_list_test)

def get_dataset():
  if not os.path.exists(os.path.join(data_dir, 'Train', 'test.txt')):
    create_dataset()
  if not os.path.exists(os.path.join(data_dir, 'cache', 'test')):
    os.makedirs(os.path.join(data_dir, 'cache', 'test'))
  with open(os.path.join(data_dir, 'Train', 'test.txt'), 'r') as test_file:
    test_list = test_file.readlines()
  test_list = [x.strip() for x in test_list]
  test_ds = tf.data.Dataset.from_tensor_slices(test_list)
  test_ds = test_ds.map(process_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  test_ds = configure_for_performance(test_ds)
  return test_ds

if __name__ == '__main__':
  get_dataset()

