import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import cv2
import numpy as np


def custom_preprocess_input(image):
    """
    Custom preprocessing function that does not alter the image.
    """
    return image/255.0
# Use custom_preprocess_input as the preprocessing function
train_datagen = ImageDataGenerator(preprocessing_function=custom_preprocess_input, validation_split=0.2)

# Rest of your code remains the same
train_generator = train_datagen.flow_from_directory(
    'ChessPiecesDataset/',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='training')



validation_generator = train_datagen.flow_from_directory(
    'ChessPiecesDataset/',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation')


base_model = InceptionV3(weights='imagenet', include_top=False)


x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(13, activation='softmax')(x) 

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=validation_generator, epochs=5)
train_accuracy = model.evaluate(train_generator)[1]
print(f'Train Set Accuracy: {train_accuracy * 100:.2f}%')
model.save('chess_piece_classifier_model.h5')


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# import cv2
# import numpy as np

# def custom_preprocess_input(image):
#     """
#     Custom preprocessing function that normalizes the image.
#     """
#     return image / 255.0

# # Use custom_preprocess_input as the preprocessing function
# train_datagen = ImageDataGenerator(preprocessing_function=custom_preprocess_input, validation_split=0.2)

# train_generator = train_datagen.flow_from_directory(
#     'ChessPiecesDataset/',
#     target_size=(299, 299),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training')

# validation_generator = train_datagen.flow_from_directory(
#     'ChessPiecesDataset/',
#     target_size=(299, 299),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation')

# # Load pre-trained ResNet50 model
# base_model = ResNet50(weights='imagenet', include_top=False)

# # Add custom layers
# x = base_model.output
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# x = tf.keras.layers.Dense(1024, activation='relu')(x)
# predictions = tf.keras.layers.Dense(13, activation='softmax')(x)

# model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(train_generator, validation_data=validation_generator, epochs=5)

# # Evaluate the model
# train_accuracy = model.evaluate(train_generator)[1]
# print(f'Train Set Accuracy: {train_accuracy * 100:.2f}%')

# # Save the model
# model.save('chess_piece_classifier_resnet_model.h5')
