import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import cv2
import numpy as np


def custom_preprocess_input(image):
    """
    Preprocesses an image by applying Laplacian filtering for edge detection.
    Ensures compatibility with JPEG images and models like InceptionV3.
    """
    # Convert to uint8 if needed
    if image.max() <= 1:
        image = (image * 255).astype('uint8')
    else:
        image = image.astype('uint8')

    # Convert to grayscale
    if image.ndim == 3 and image.shape[-1] == 3:  # If RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)  # Apply edge detection
    laplacian = np.absolute(laplacian)  # Take absolute values
    laplacian = laplacian / laplacian.max()  # Normalize to [0, 1]

    # Convert back to 3 channels (InceptionV3 requires 3-channel input)
    laplacian = np.stack([laplacian] * 3, axis=-1)

    return laplacian


# Create ImageDataGenerator with custom preprocessing
train_datagen = ImageDataGenerator(preprocessing_function=custom_preprocess_input, validation_split=0.2)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    'ChessPiecesDataset/',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    'ChessPiecesDataset/',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load InceptionV3 base model (without top layers)
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(13, activation='softmax')(x)  # 12 pieces + empty square

# Create the complete model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=validation_generator, epochs=5)

# Evaluate the model
train_accuracy = model.evaluate(train_generator)[1]
print(f'Train Set Accuracy: {train_accuracy * 100:.2f}%')

# Save the trained model
model.save('chess_piece_classifier_inceptionv3.h5')
