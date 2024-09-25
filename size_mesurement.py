import cv2
import numpy as np
class scale_measurement:
    def __init__(self):
        self.real_height_mm = 85.6  
        self.real_width_mm = 53.98
    @staticmethod
    def calculate_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def order_points(self,pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def find_largest_rectangle(self,image):
        rectangles = []
        blurred = cv2.medianBlur(image, 9)
        for c in range(3):
            ch = cv2.split(blurred)[c]
            for l in range(2):
                if l == 0:
                    gray = cv2.Canny(ch, 50, 150, apertureSize=3)
                    gray = cv2.dilate(gray, None)
                else:
                    retval, gray = cv2.threshold(ch, (l + 1) * 255 // 2, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    epsilon = cv2.arcLength(contour, True) * 0.02
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4 and cv2.contourArea(approx) > 1000 and cv2.isContourConvex(approx):
                        rectangles.append(approx)

        if len(rectangles) > 0:
            largest_rectangle = max(rectangles, key=cv2.contourArea)
            return self.order_points(largest_rectangle.reshape(4, 2))
        else :
            print('Could not detect rectangle')
            return None

    def calculate_zoom_factors(self,measured_height, measured_width):
        zoom_factor_height = measured_height / self.real_height_mm
        zoom_factor_width = measured_width / self.real_width_mm
        return zoom_factor_height, zoom_factor_width

    def get_scale_factor(self,image_path):
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Could not load image.")
            return None
        reference_rectangle = self.find_largest_rectangle(image)
        if reference_rectangle is None:
            print("Error: Could not detect reference object.")
            average_zoom_factor=1
            mm_per_pix=1
        else : 
            top_left, top_right, bottom_right, bottom_left = reference_rectangle
            print(top_left, top_right, bottom_right, bottom_left)
            measured_width_px = self.calculate_distance(top_left, top_right)
            measured_height_px = self.calculate_distance(top_left, bottom_left)

            zoom_factor_height, zoom_factor_width = self.calculate_zoom_factors(measured_height_px, measured_width_px)
            
            average_zoom_factor = (float(zoom_factor_height) + float(zoom_factor_width)) / 2
            mm_per_pix=1/float(average_zoom_factor)
            print(f"Measured width in image: {measured_width_px:.2f} pixels")
            print(f"Measured height in image: {measured_height_px:.2f} pixels")
            print(f"Real-world width of card: {self.real_width_mm} mm")
            print(f"Real-world height of card: {self.real_height_mm} mm")
            print(f"Zoom Factor (Height): {zoom_factor_height:.2f}")
            print(f"Zoom Factor (Width): {zoom_factor_width:.2f}")
            print(f"mm per pixel: {mm_per_pix:.2f}")
            cv2.drawContours(image, [reference_rectangle.astype(int)], -1, (0, 255, 0), 2)
            cv2.imwrite("detected_rectangle.jpg", image)
            cv2.imshow("Detected Rectangle", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return float(average_zoom_factor)

    def convert_to_mm(self,measurements_dict,average_zoom_factor):
        mm_per_pix=1/float(average_zoom_factor)
        for key, value in measurements_dict.items():
            measurements_dict[key] = value * mm_per_pix
        return measurements_dict


def test_scale_measurement(image_path):
    # Instantiate the scale_measurement class
    scale_measure = scale_measurement()
    
    # Define the real-world dimensions of the reference object
    real_height_mm = 85.6
    real_width_mm = 53.98
    
    # Call the get_scale_factor method
    average_zoom_factor = scale_measure.get_scale_factor(image_path, real_height_mm, real_width_mm)
    
    # Print the result
    if average_zoom_factor is not None:
        print(f"Average Zoom Factor: {average_zoom_factor:.2f}")
    else:
        print("Failed to calculate the zoom factor.")

if __name__ == "__main__":
    # Path to the test image
    image_path = r"C:\Users\ayabe\Downloads\456861485_1175898570363681_953152550565035295_n.jpg"
    
    # Run the test
    test_scale_measurement(image_path)