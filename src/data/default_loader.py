from os.path import join
from hydra import experimental
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from itertools import islice
import os
from PIL import Image
import numpy as np
import tensorflow.keras.layers as layers
from time import time
from omegaconf import DictConfig, OmegaConf
import os

from data.config_utils import find_dataset_folder
from data.dataset_loader_base import DataLoaderBase
from data.loaded_dataset import LoadedDataset


def get_preprocessing_layers(ds_mean):
    return tf.keras.Sequential(
        [
            layers.experimental.preprocessing.Rescaling(1.0/255.0),
            layers.experimental.preprocessing.Normalization(mean=ds_mean, variance=1.0),
        ]
    )


class FaceDsLoader(DataLoaderBase):
    def __init__(self, save_dir1: str='../__dataset/serena/',save_dir2: str='../__dataset/novak/', dataset_config: DictConfig = None):
        self.save_dir1 = save_dir1
        self.save_dir2 = save_dir2
        self.has_validation = False
        if dataset_config is not None:
            self.has_validation = True if dataset_config.get('has_validation', None) is not None and dataset_config.get('has_validation') == True else False

    def to_original(self, img):
        org = img + self.ds_mean
        return org

    def __adjust_save_dir_to_cwd(self):
        self.save_dir1 = self.save_dir1.replace('/', os.path.sep)
        self.save_dir2 = self.save_dir2.replace('/', os.path.sep)

        self.save_dir1 = find_dataset_folder(self.save_dir1)
        self.save_dir2 = find_dataset_folder(self.save_dir2)
        
        print('save_dir1: ', self.save_dir1)
        print('save_dir2: ', self.save_dir2)


    def __split_training(self, cfg, p1_train_dir, p2_train_dir, p1_train_ds):
        if cfg['training']['val_batch_size'] is None:
            val_ds_cardinality = int(p1_train_ds.cardinality().numpy()*p1_train_ds._batch_size.numpy() * cfg['training']['val_split'])
            cfg['training']['val_batch_size'] = val_ds_cardinality
        p1_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                p1_train_dir, 
                label_mode=None, class_names=None, color_mode='rgb', image_size=(96,96), 
                batch_size=cfg['training']['val_batch_size'], shuffle=False,
                interpolation='bilinear', validation_split=cfg['training']['val_split'],
                subset=None if cfg['training']['val_split'] == 0 else "validation",
                seed=self.p1_seed
            )
        p2_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                p2_train_dir, 
                label_mode=None, class_names=None, color_mode='rgb', image_size=(96,96), 
                batch_size=cfg['training']['val_batch_size'], shuffle=False,
                interpolation='bilinear', validation_split=cfg['training']['val_split'], 
                subset=None if cfg['training']['val_split'] == 0 else "validation",
                seed=self.p2_seed
            )
        return p1_val_ds, p2_val_ds

    def __call__(self, cfg: DictConfig) -> LoadedDataset:        
        self.__adjust_save_dir_to_cwd()

        p1_train_dir = os.path.join(self.save_dir1, 'training')
        p2_train_dir = os.path.join(self.save_dir2, 'training') 
        p1_val_dir = os.path.join(self.save_dir1, 'validation')
        p2_val_dir = os.path.join(self.save_dir2, 'validation')
        print('p1 train dir: ', p1_train_dir)
        print('p2 train dir: ', p2_train_dir)

        p1_test_dir = os.path.join(self.save_dir1, 'test')
        p2_test_dir = os.path.join(self.save_dir2, 'test')

        self.p1_seed = cfg['training']['seed']
        self.p2_seed = cfg['training']['seed']

        p1_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            p1_train_dir,
            label_mode=None, class_names=None, color_mode='rgb', image_size=(96,96), 
            batch_size=cfg['training']['batch_size'], shuffle=True, seed=self.p1_seed,
            interpolation='bilinear', validation_split=cfg['training']['val_split'], 
            subset=None if cfg['training']['val_split']== 0 or cfg['training']['val_split'] is None else "training"
        )
        p2_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            p2_train_dir, label_mode=None,
            class_names=None, color_mode='rgb', image_size=(96,96), 
            batch_size=cfg['training']['batch_size'], shuffle=True, seed=self.p2_seed,
            interpolation='bilinear', validation_split=cfg['training']['val_split'], 
            subset=None if cfg['training']['val_split']== 0 or cfg['training']['val_split'] is None else "training"
        )

        if cfg['training']['val_split'] > 0:
            assert self.has_validation == False
            p1_val_ds, p2_val_ds = self.__split_training(cfg, p1_train_dir, p2_train_dir, p1_train_ds)
        elif self.has_validation:
            if cfg['training']['val_batch_size'] is None or cfg['training']['val_batch_size'] == 0:
                #assuming lenghts are equal
                cfg['training']['val_batch_size'] = len(os.listdir(os.path.join(p1_val_dir, "images")))
            p1_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                p1_val_dir,
                label_mode=None, class_names=None, color_mode='rgb', image_size=(96,96), 
                batch_size=cfg['training']['val_batch_size'], shuffle=False, seed=self.p1_seed,
                interpolation='bilinear'
            )
            p2_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
                p2_val_dir, label_mode=None,
                class_names=None, color_mode='rgb', image_size=(96,96), 
                batch_size=cfg['training']['val_batch_size'], shuffle=False, seed=self.p2_seed,
                interpolation='bilinear'
            )
        else:
            p1_val_ds, p2_val_ds = None, None


        if os.path.exists(p1_test_dir) and os.path.exists(p2_test_dir):
            test_len = len(os.listdir(os.path.join(p1_test_dir, "images")))
            if cfg['training']['test_batch_size'] is None:
                cfg['training']['test_batch_size'] = test_len

            p1_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    p1_test_dir, 
                    label_mode=None, class_names=None, color_mode='rgb', image_size=(96,96), 
                    batch_size=cfg['training']['test_batch_size'], shuffle=False, seed=self.p1_seed,
                    interpolation='bilinear'
            )
            p2_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
                    p2_test_dir, 
                    label_mode=None, class_names=None, color_mode='rgb', image_size=(96,96), 
                    batch_size=cfg['training']['test_batch_size'], shuffle=False, seed=self.p2_seed,
                    interpolation='bilinear'
            )
        else:
            p1_test_ds, p2_test_ds = None, None

        i1 = os.path.join(p1_train_dir, 'images')
        i2 = os.path.join(p2_train_dir, 'images')
        i1_files = os.listdir(i1) if cfg['training']['take_batch'] is None else islice(os.listdir(i1), cfg['training']['take_batch'] * cfg['training']['batch_size'])
        i2_files = os.listdir(i2) if cfg['training']['take_batch'] is None else islice(os.listdir(i2), cfg['training']['take_batch'] * cfg['training']['batch_size'])

        full_ds_p1 = [Image.open(f) for f in [os.path.join(i1,i) for i in i1_files]]
        full_ds_p1 = [np.asarray( img, dtype="uint32" ) for img in full_ds_p1]
        full_ds_p2 = [Image.open(f) for f in [os.path.join(i2,i) for i in i2_files]]
        full_ds_p2 = [np.asarray( img, dtype="uint32" ) for img in full_ds_p2]
        full_ds = full_ds_p1 + full_ds_p2

        prev_p1 = []
        prev_p2 = []
        total_prev = 3


        # div by 255.0 cause of further scalling in preprocess layer
        self.ds_mean = np.mean(full_ds, axis=(0,1,2), dtype=np.float32)/255.0

        if len(full_ds_p1) < total_prev:
            raise ValueError("Training dataset does not contain at most 3 items")
        if len(full_ds_p2) < total_prev:
            raise ValueError("Training dataset does not contain at most 3 items")
        
        ds_to_prev_p1 = p1_val_ds if p1_val_ds is not None else p1_test_ds
        ds_to_prev_p2 = p2_val_ds if p2_val_ds is not None else p2_test_ds

        rand_tr = tf.random.uniform([1], 0, len(full_ds_p1), tf.int32).numpy()[0]
        prev_p1.append(np.expand_dims(full_ds_p1[rand_tr], axis=0))
        for x in ds_to_prev_p1:
            for i in range(x.shape[0]):
                prev_p1.append(np.expand_dims(x[i], axis=0))
                if len(prev_p1) == total_prev:
                    break
            if len(prev_p1) == total_prev:
                break
        prev_p2.append(np.expand_dims(full_ds_p2[rand_tr], axis=0))
        for x in ds_to_prev_p2:
            for i in range(x.shape[0]):
                prev_p2.append(np.expand_dims(x[i], axis=0))
                if len(prev_p2) == total_prev:
                    break
            if len(prev_p2) == total_prev:
                break

        
        p1_train_ds = p1_train_ds.map(lambda x: (x, x/255.0), num_parallel_calls=tf.data.AUTOTUNE)
        p2_train_ds = p2_train_ds.map(lambda x: (x, x/255.0), num_parallel_calls=tf.data.AUTOTUNE)

        if(cfg['training']['take_batch'] is not None):
            p1_train_ds = p1_train_ds.take(cfg['training']['take_batch'])
            p2_train_ds = p2_train_ds.take(cfg['training']['take_batch'])

        if p1_val_ds is not None and p2_val_ds is not None:
            p1_val_ds = p1_val_ds.map(lambda x: (x, x/255.0), num_parallel_calls=tf.data.AUTOTUNE)
            p2_val_ds = p2_val_ds.map(lambda x: (x, x/255.0), num_parallel_calls=tf.data.AUTOTUNE)
            p1_val_ds = p1_val_ds.prefetch(tf.data.AUTOTUNE)
            p2_val_ds = p2_val_ds.prefetch(tf.data.AUTOTUNE)
        
        if p1_test_ds is not None and p2_test_ds is not None:
            p1_test_ds = p1_test_ds.map(lambda x: (x, x/255.0), num_parallel_calls=tf.data.AUTOTUNE)
            p2_test_ds = p2_test_ds.map(lambda x: (x, x/255.0), num_parallel_calls=tf.data.AUTOTUNE)
            p1_test_ds = p1_test_ds.prefetch(tf.data.AUTOTUNE)
            p2_test_ds = p2_test_ds.prefetch(tf.data.AUTOTUNE)

        p1_train_ds = p1_train_ds.prefetch(tf.data.AUTOTUNE)
        p2_train_ds = p2_train_ds.prefetch(tf.data.AUTOTUNE)
        

        return LoadedDataset(p1_train_ds, p2_train_ds, p1_val_ds, p2_val_ds, p1_test_ds, p2_test_ds, prev_p1, prev_p2)

