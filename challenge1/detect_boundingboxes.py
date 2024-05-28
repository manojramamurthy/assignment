import cv2
import os
import pandas as pd

# This will convert bounding boxes to yolo format
def convert_to_yolo_format(x_min, y_min, x_max, y_max, image_width, image_height, class_id):
    """
    Converts bounding box coordinates to YOLO format.
    
    Args:
        x_min (float): Leftmost x-coordinate of the bounding box.
        y_min (float): Topmost y-coordinate of the bounding box.
        x_max (float): Rightmost x-coordinate of the bounding box.
        y_max (float): Bottommost y-coordinate of the bounding box.
        image_width (float): Width of the image.
        image_height (float): Height of the image.
        class_id (int): Class ID of the object.
        
    Returns:
        list: YOLO format representation of the bounding box [class_id, x_center_normalized, y_center_normalized, width_normalized, height_normalized].
    """
    
    # Calculate the width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min
    
    # Calculate the center coordinates of the bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Normalize the center coordinates and the width and height
    x_center_normalized = x_center / image_width
    y_center_normalized = y_center / image_height
    width_normalized = width / image_width
    height_normalized = height / image_height
    
    # Create the YOLO format representation
    yolo_format = [int(class_id), x_center_normalized, y_center_normalized, width_normalized, height_normalized]
    
    return yolo_format

# Finds the coins using Open-CV and gets the bounding boxes
def get_bboxes(input_dir, file_name, output_dir):
    path = os.path.join(input_dir, file_name)
    image = cv2.imread(path)
    image = cv2.medianBlur(image,5)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    # Detect circles using the Hough Circle Transform
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is not None:
        circles = circles[0,:]

        df_bboxes = pd.DataFrame(columns=['class', 'xmin', 'ymin', 'xmax', 'ymax'])
        for i, circle in enumerate(circles):
            x, y, r = circle.astype(int)
            if x > 200 and y > 150:
                # Extract the coin from the image
                # converted =convert_bboxes_to_yolo([y - r, y + r, x - r, x + r], img_copy.shape[1], img_copy.shape[0])
                x_min = y - r
                y_min = y + r
                x_max = x - r
                y_max = x + r
                image_width = image.shape[1]
                image_height = image.shape[0]
                class_id = 0
                
                yolo_format = convert_to_yolo_format(x_min, y_min, x_max, y_max, image_width, image_height, class_id)

                df_bboxes.loc[len(df_bboxes)] = yolo_format
                df_bboxes['class'] = df_bboxes['class'].astype('int')
                
        file = os.path.join(output_dir, file_name.replace('.jpg', '.txt')) 
        df_bboxes.to_csv(file, index=False, header=False, sep=' ')


# Directory containing the input images
# input_dir = "data/coin-dataset/train/images"
# input_dir = "data/coin-dataset/val/images"
input_dir = "data/coin-dataset/test/images"

# Directory to save the extracted coins
# output_dir = "data/coin-dataset/train/labels"
# output_dir = "data/coin-dataset/val/labels"
output_dir = "data/coin-dataset/test/labels"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over the input images
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_dir, filename)
        print(image_path)
        get_bboxes(input_dir, filename, output_dir)