import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs found: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found. Ensure CUDA and cuDNN are installed and configured correctly.")

# Constants
img_size = 224
num_landmarks = 6
batch_size = 64
epochs = 25

# Load data
df = pd.read_csv(r"C:\Users\ayabe\Downloads\type1_landmarks.csv")
print(f"Total samples: {len(df)}")

# Extract image paths and keypoints
image_paths = df.iloc[:, 0].values
keypoints = df.iloc[:, 1:].values

# Data augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Custom data generator with augmentation
def data_generator(image_paths, keypoints, batch_size=32, augment=True):
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

# Model definition
def create_model(input_shape, num_keypoints):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer=glorot_uniform()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout added

    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer=glorot_uniform()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout added

    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer=glorot_uniform()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))  # Dropout added

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=glorot_uniform()))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))  # Dropout added

    model.add(Dense(64, kernel_initializer=glorot_uniform()))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(12, kernel_initializer=glorot_uniform()))

    return model

# Early stopping and checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Custom callback for progress display
class ProgressCallback(Callback):
    def __init__(self, num_epochs, num_steps):
        super().__init__()
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.epoch_progress_bar = tqdm(total=num_epochs, position=0, desc="Epochs")
        self.step_progress_bar = tqdm(total=num_steps, position=1, desc="Steps")

    def on_epoch_begin(self, epoch, logs=None):
        self.step_progress_bar.reset()

    def on_batch_end(self, batch, logs=None):
        self.step_progress_bar.update(1)
        self.step_progress_bar.set_postfix({
            'loss': f"{logs['loss']:.4f}",
            'mse': f"{logs['mean_squared_error']:.4f}"
        })

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress_bar.update(1)
        self.epoch_progress_bar.set_postfix({
            'loss': f"{logs['loss']:.4f}",
            'val_loss': f"{logs['val_loss']:.4f}",
            'mse': f"{logs['mean_squared_error']:.4f}",
            'val_mse': f"{logs['val_mean_squared_error']:.4f}"
        })

    def on_train_end(self, logs=None):
        self.epoch_progress_bar.close()
        self.step_progress_bar.close()

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_num = 1
input_shape = (img_size, img_size, 3)

for train_idx, val_idx in kf.split(image_paths):
    print(f"\nTraining fold {fold_num}...")
    
    # Split data for this fold
    train_paths, val_paths = image_paths[train_idx], image_paths[val_idx]
    train_keypoints, val_keypoints = keypoints[train_idx], keypoints[val_idx]
    
    # Create model
    model = create_model(input_shape, num_landmarks)
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['mean_squared_error'])

    # Generators
    train_generator = data_generator(train_paths, train_keypoints, batch_size=batch_size, augment=True)
    val_generator = data_generator(val_paths, val_keypoints, batch_size=batch_size, augment=False)

    # Fit model
    steps_per_epoch = len(train_paths) // batch_size
    validation_steps = len(val_paths) // batch_size
    progress_callback = ProgressCallback(epochs, steps_per_epoch)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping, progress_callback],
        verbose=0
    )

    print(f"Fold {fold_num} training complete.\n")
    fold_num += 1

# Save final model
model.save('final_upper_model.h5')
print("Final model saved as 'final_upper_model.h5'")
