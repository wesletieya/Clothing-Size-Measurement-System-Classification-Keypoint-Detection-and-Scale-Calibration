import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load CSVs and create DataFrame
def load_and_label(csv_path, label):
    df = pd.read_csv(csv_path)
    df = df[['image_path']]
    df['label'] = label
    return df

csv_class_1 = load_and_label(r"C:\Users\ayabe\Downloads\type1_landmarks.csv", 0)
csv_class_2 = load_and_label(r"C:\Users\ayabe\Downloads\type2_landmarks.csv", 1)
csv_class_3 = load_and_label(r"C:\Users\ayabe\Downloads\type3_landmarks.csv", 2)

df = pd.concat([csv_class_1, csv_class_2, csv_class_3], ignore_index=True)

# Sample 1000 random rows from the DataFrame
df_sampled = df.sample(n=3000, random_state=42)  # random_state ensures reproducibility

# Image preprocessing function
def preprocess_image(image_path, img_size=(224, 224)):
    img = load_img(image_path, target_size=img_size)
    img = img_to_array(img)
    img = img / 255.0  # Rescale pixel values between 0 and 1
    return img

# Batch processing for image preprocessing
def preprocess_images_batch(image_paths, batch_size=32, img_size=(224, 224)):
    images = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [preprocess_image(path, img_size=img_size) for path in batch_paths]
        images.extend(batch_images)
    return np.array(images)

# Prepare images for prediction in batches
image_paths = df_sampled['image_path'].values
images = preprocess_images_batch(image_paths)

# Load model
model = tf.keras.models.load_model(r'C:\Users\ayabe\vs projects\finetuned_classification2.h5')

# Ensure the model is in eager execution mode for debugging
tf.config.run_functions_eagerly(True)

# Batch prediction function
def batch_predict(model, images, batch_size=8):
    predictions = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # Check for empty batches
        if batch.size == 0:
            print(f'Skipping empty batch at index {i}')
            continue
        
        # Ensure the batch has the correct shape
        print(f'Processing batch {i//batch_size + 1}, batch shape: {batch.shape}')
        
        batch_predictions = model.predict(batch)
        
        if batch_predictions.size == 0:
            print(f'Warning: Batch {i//batch_size + 1} resulted in empty predictions')
        else:
            predictions.append(batch_predictions)
    
    return np.vstack(predictions) if predictions else np.array([])

# Call batch_predict
predictions = batch_predict(model, images)

# Ensure there are valid predictions
if predictions.size > 0:
    # Get predicted class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Add predictions to DataFrame
    df_sampled['predicted_label'] = predicted_labels

    # Prepare for further training
    X_train, X_val, y_train, y_val = train_test_split(images, df_sampled['label'].values, test_size=0.2, random_state=42)

    # Compile and train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_val, y_val))

    # Save the updated model
    model.save('finetuned_classification3.h5')
else:
    print("No valid predictions were made.")
