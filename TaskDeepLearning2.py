# Name: Eiad Ahmed Mohamed Zaki Elemam
# Task2: DeepLearning2 - CNN Implementation on CIFAR-10 Dataset

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

train_labels_cat = to_categorical(train_labels, 10)
test_labels_cat = to_categorical(test_labels, 10)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
EPOCHS = 10
BATCH_SIZE = 64

history = model.fit(
    train_images, train_labels_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(test_images, test_labels_cat)
)
test_loss, test_acc = model.evaluate(test_images, test_labels_cat, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")