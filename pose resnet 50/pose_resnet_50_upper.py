import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load CSV file
csv_file = r"C:\Users\ayabe\Downloads\type1_landmarks.csv"
data = pd.read_csv(csv_file)

# Verify columns
print("Columns in the CSV file:", data.columns)
print("Number of columns in the CSV file:", len(data.columns))

# Define data loading function for the dataset
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image = cv2.resize(image, (224, 224))  # Resize image to match input shape
    image = image / 255.0  # Normalize pixel values
    return image

def preprocess_landmarks(row):
    landmarks = []
    num_landmarks = 6  # Number of landmarks

    for i in range(num_landmarks):
        x_col_index = i * 2 + 1  # Column index for x location
        y_col_index = i * 2 + 2  # Column index for y location
        
        if x_col_index >= len(row) or y_col_index >= len(row):
            raise IndexError(f"Index out of bounds: x_col_index={x_col_index}, y_col_index={y_col_index}")

        x = row[x_col_index]
        y = row[y_col_index]

        landmarks.extend([x, y])
    
    return np.array(landmarks, dtype=np.float32)

# Create a TensorFlow dataset from CSV data
def data_generator():
    for _, row in data.iterrows():
        try:
            img = load_image(row[0])  # Assuming image path is in the first column
            lm = preprocess_landmarks(row)
            yield img, lm
        except Exception as e:
            print(f"Error processing row: {e}")

# Create a dataset with batch size
batch_size = 16
dataset = tf.data.Dataset.from_generator(data_generator,
                                         output_signature=(
                                             tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                                             tf.TensorSpec(shape=(12,), dtype=tf.float32)
                                         ))
dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Define Pose ResNet-50 Model
def build_pose_resnet50_model(input_shape, num_landmarks):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    output = tf.keras.layers.Dense(num_landmarks * 2)(x)  # num_landmarks * 2 for x, y coordinates
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Custom callback to display training progress
class TrainingProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}:")
        print(f" - Loss: {logs.get('loss'):.4f}")
        print(f" - Validation Loss: {logs.get('val_loss'):.4f}")

# Build and compile model
input_shape = (224, 224, 3)  # Example input shape, adjust as needed
num_landmarks = 6  # Number of landmarks
model = build_pose_resnet50_model(input_shape, num_landmarks)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error')

# Set up callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('pose_resnet50_model_upper.h5', 
                                                        save_best_only=True, 
                                                        monitor='val_loss', 
                                                        mode='min', 
                                                        verbose=1)
progress_callback = TrainingProgress()

history = model.fit(dataset, 
                    epochs=10, 
                    validation_data=dataset,
                    callbacks=[checkpoint_callback, progress_callback])

print("Training complete.")
