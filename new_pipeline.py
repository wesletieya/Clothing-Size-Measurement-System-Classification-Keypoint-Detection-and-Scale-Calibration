from size_mesurement import *
from functions import *



image_path=r"C:\Users\ayabe\Downloads\456861485_1175898570363681_953152550565035295_n.jpg"
imag=Image(image_path,224)
img_array, image,img, h, w = imag.load_image()
get_category=get_category()
category=get_category.category(img_array)
print('the category is : ' ,category)
if category!=0:
    image=Image(image_path,96)
    img_array, image,img, h, w = imag.load_image()
landmarks=Landmarks()
predicted_landmarks=landmarks.predict_landmarks(category,img_array)
scaled_landmarks=landmarks.scale_landmarks(predicted_landmarks,w,h)
measurements=Measurements()
measrements_dict=measurements.measure_distance(category,scaled_landmarks)
print('the measurements are in pixels are: ' ,measrements_dict)

scale=scale_measurement()
average_zoom_factor=scale.get_scale_factor(image_path)
measrements_in_mm=scale.convert_to_mm(measrements_dict,average_zoom_factor)
print('the measurements are in mm are: ' ,measrements_in_mm)




