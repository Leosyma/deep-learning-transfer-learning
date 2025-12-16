# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 16:06:54 2025

@author: Leonardo
"""

#%% Bibliotecas
import numpy as np
import cv2
import PIL.Image as Image
import os
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

import tf_keras  # <- Keras 2 compatível com TF Hub
from tf_keras import layers, models


#%% Importando o modelo treinado do hub
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

#%% Carrega o dataset de flores
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='.', untar=True)

data_dir


import pathlib
data_dir = pathlib.Path(data_dir)
data_dir

list(data_dir.glob('*/*/*.jpg'))[:5]
image_count = len(list(data_dir.glob('*/*/*.jpg')))
print(image_count)


roses = list(data_dir.glob('flower_photos/roses/*'))
roses[:5]
Image.open(str(roses[1]))

tulips = list(data_dir.glob('flower_photos/tulips/*'))
Image.open(str(tulips[0]))

#%% Converte em numpy array
flowers_images_dict = {
    'roses': list(data_dir.glob('flower_photos/roses/*')),
    'daisy': list(data_dir.glob('flower_photos/daisy/*')),
    'dandelion': list(data_dir.glob('flower_photos/dandelion/*')),
    'sunflowers': list(data_dir.glob('flower_photos/sunflowers/*')),
    'tulips': list(data_dir.glob('flower_photos/tulips/*')),
}

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}

X, y = [], []

for flower_name, images in flowers_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(flowers_labels_dict[flower_name])
        
X = np.array(X)
y = np.array(y)     

#%% Separa em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)        

#%% Normaliza os dados
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

#%% Treina o modelo
num_of_flowers = 5

model = models.Sequential([
  pretrained_model_without_top_layer,
  layers.Dense(num_of_flowers)
])

model.summary()

model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=5)

model.evaluate(X_test_scaled,y_test)
