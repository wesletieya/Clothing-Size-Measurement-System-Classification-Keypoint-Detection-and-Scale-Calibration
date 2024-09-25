import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import matplotlib.pyplot as plt

# Constants
img_size = 224
batch_size = 100
epochs = 30

# Load and preprocess data
data = pd.read_csv(r"C:\Users\ayabe\Downloads\type3_landmarks.csv")
# Print information about the dataset
print(f"Number of samples: {len(data)}")
print(f"Number of columns: {len(data.columns)}")
print(f"Column names: {data.columns.tolist()}")

image_paths = [os.path.join('images', img) for img in data['image_path'].values]
keypoints = data.drop('image_path', axis=1).values

# Print shape of keypoints
print(f"Shape of keypoints: {keypoints.shape}")

# Calculate the number of keypoints
num_keypoints = keypoints.shape[1]
print(f"Number of keypoints: {num_keypoints}")

# Split data
train_paths, val_paths, train_keypoints, val_keypoints = train_test_split(
    image_paths, keypoints, test_size=0.2, random_state=42
)

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Custom data generator with augmentation
def data_generator(image_paths, keypoints, batch_size=64, augment=True):
    while True:
        for offset in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[offset:offset+batch_size]
            batch_keypoints = keypoints[offset:offset+batch_size]
            batch_images = []

            for path in batch_paths:
                img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
                img_array = img_to_array(img)
                img_array = img_array / 255.0  # Normalize

                if augment:
                    img_array = data_gen.random_transform(img_array)  # Apply augmentation

                batch_images.append(img_array)

            yield np.array(batch_images), np.array(batch_keypoints)

# Create model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_keypoints)  # Adjusted to match the actual number of keypoints
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Print model summary
model.summary()

# Train model
train_generator = data_generator(train_paths, train_keypoints, batch_size)
val_generator = data_generator(val_paths, val_keypoints, batch_size, augment=False)

steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(val_paths) // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=validation_steps,
    verbose=1
)

# Save final model
model.save('full.keras', save_format='keras')
print("Final model saved as 'full_model.h5'")

