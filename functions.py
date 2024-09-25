
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Image:
    def __init__(self, image_path,shape):
        self.image_path = image_path
        self.shape=(shape,shape,3)

    def load_image(self):
        image = cv2.imread(self.image_path)
        h, w, _ = image.shape
        img = load_img(self.image_path, color_mode='rgb', target_size=self.shape[:2])
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array, image,img, h, w
    
    def visualize_predictions(self,image, predicted_keypoints, save_path):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
        # Plot predicted keypoints
        plt.scatter(predicted_keypoints[::2], predicted_keypoints[1::2], c='r', marker='x', s=100, label='Predicted Keypoints')
        
        for i, (x, y) in enumerate(zip(predicted_keypoints[::2], predicted_keypoints[1::2])):
            plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points', color='white', fontsize=12, fontweight='bold')
        
        plt.legend()
        plt.title('Predicted Keypoints')
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()


class get_category:
    def __init__(self):
        self.model_path=r'stage\new_new\working\finetuned_classification3.h5'
        self.model=load_model(self.model_path)

    def category(self,image):
        category=self.model.predict(image)
        category=np.argmax(category)
        return category
    
class Measurements:
    def __init__(self):
        pass 

    def load_csv(self):
        df=pd.read_csv(self.csv_path)
        return df
    
    def calculate_distance(self,point1,point2):
        x1,y1=point1
        x2,y2=point2
        distance=np.sqrt((x2-x1)**2+(y2-y1)**2)
        return distance
    

    def measurements_upper_body(self,landmarks):
        left_collar = landmarks[0:2]
        right_collar = landmarks[2:4]
        left_sleeve = landmarks[4:6]
        right_sleeve = landmarks[6:8]
        left_hem = landmarks[8:10]
        right_hem = landmarks[10:]
        collar_width = self.calculate_distance(left_collar, right_collar)
        left_shoulder_width = self.calculate_distance(left_sleeve, left_collar)
        right_shoulder_width = self.calculate_distance(right_sleeve, right_collar)
        shoulder_width = (left_shoulder_width + right_shoulder_width) / 2
        length = (self.calculate_distance(left_collar, left_hem)+self.calculate_distance(right_collar, right_hem))/2
        waist_width = self.calculate_distance(left_hem, right_hem)
        length = (self.calculate_distance(left_collar, left_hem) + self.calculate_distance(right_collar, right_hem)) / 2
        return collar_width,shoulder_width,length,waist_width,length
    

    def measurements_lower_body(self,landmarks):
        left_waist = landmarks[0:2]
        right_waist = landmarks[2:4]
        left_hem = landmarks[4:6]
        right_hem = landmarks[6:]
        waist_width = self.calculate_distance(left_waist, right_waist)
        length = (self.calculate_distance(left_waist, left_hem) + self.calculate_distance(right_waist, right_hem)) / 2
        return waist_width,length
    

    def measurements_full_body(self,landmarks):
        left_collar = landmarks[0:2]
        right_collar = landmarks[2:4]
        left_sleeve = landmarks[4:6]
        right_sleeve = landmarks[6:8]
        left_waist = landmarks[8:10]
        right_waist = landmarks[10:12]
        left_hem = landmarks[12:14]
        right_hem = landmarks[14:]
        collar_width = self.calculate_distance(left_collar, right_collar)
        left_shoulder_width = self.calculate_distance(left_sleeve, left_collar)
        right_shoulder_width = self.calculate_distance(right_sleeve, right_collar)
        full_shoulder_width = self.calculate_distance(left_sleeve, right_sleeve)
        shoulder_width = (left_shoulder_width + right_shoulder_width) / 2
        bottom_width = self.calculate_distance(left_hem, right_hem)
        waist_width = self.calculate_distance(left_waist, right_waist)
        length = (self.calculate_distance(left_collar, left_hem) + self.calculate_distance(right_collar, right_hem)) / 2
        top_length = (self.calculate_distance(left_waist, left_collar) + self.calculate_distance(right_waist, right_collar)) / 2
        bottom_length = (self.calculate_distance(left_waist, left_hem) + self.calculate_distance(right_waist, right_hem)) / 2
        return collar_width,shoulder_width,full_shoulder_width,waist_width,bottom_width,length,top_length,bottom_length
   
    
    def measure_distance(self, category, landmarks):
        measurements = {}
        if category == 0:
            (measurements['collar width'], measurements['shoulder width'],
             measurements['full shoulder width'], measurements['waist width'],
             measurements['length']) = self.measurements_upper_body(landmarks)
        elif category == 1:
            measurements['waist width'], measurements['length'] = self.measurements_lower_body(landmarks)
        elif category == 2:
            (measurements['collar width'], measurements['shoulder width'],
             measurements['full shoulder width'], measurements['waist width'],
             measurements['bottom width'], measurements['length'],
             measurements['top length'], measurements['bottom length']) = self.measurements_full_body(landmarks)
        return measurements
class Landmarks:
    def __init__(self):
        self.upper_path=r'stage\new_new\working\upper.h5'
        self.lower_path=r'stage\new_new\working\lower_model.h5'
        self.full_path=r'stage\new_new\working\full_model.h5'


    def predict_landmarks(self,category,image):
        if category==0:
            model=load_model(self.upper_path)
        elif category==1:
            model=load_model(self.lower_path)
        elif category==2:
            model=load_model(self.full_path)
        landmarks = model.predict(image)
        if landmarks.ndim > 1:
            landmarks = landmarks[0]  # Handle batch dimension if present
        for i in range(0, len(landmarks), 2):
            print(f"Keypoint {i//2 + 1}: ({landmarks[i]:.2f}, {landmarks[i+1]:.2f})")
        return np.clip(landmarks,0,224).astype(int)

    def scale_landmarks(self,landmarks,original_width,original_height):
        landmarks = np.array(landmarks)
        y_landmarks = (landmarks[::2] * original_width)/224
        x_landmarks = (landmarks[1::2] * original_height)/224
        return [int(x) for pair in zip(x_landmarks, y_landmarks) for x in pair]
