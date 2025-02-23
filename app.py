from flask import Flask, request, render_template
import cv2
import numpy as np
import os
from ultralytics import YOLO
import json
import datetime
import uuid
import math
import cvzone
from PIL import Image


app = Flask(__name__)

model_paths = {
    'l-bestmodel': 'static/models/l-best.pt',
    'm-bestmodel': 'static/models/m-best.pt',
    'n-bestmodel': 'static/models/n-best.pt',
    'x-bestmodel': 'static/models/x-best.pt',
    's-bestmodel': 'static/models/s-best.pt',
}

# Load class names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

def save_image_with_bounding_boxes(img, detections, file_path):
    for detection in detections:
        if len(detection) < 6:
            continue  # Skip this detection if it does not have enough elements

        x1, y1, x2, y2, class_idx, conf = detection
        x1, y1, x2, y2, class_idx, conf = int(x1), int(y1), int(x2), int(y2), int(class_idx), int(conf)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 9, (190, 200, 100), cv2.FILLED)
        cvzone.cornerRect(img, [x1, y1, w, h], rt=1)
        
        class_name = classnames[class_idx]  # Use class_idx here
        cvzone.putTextRect(img, f'{class_name}', [x1 + 10, y1 + 10], scale=1.2, thickness=2)
    cv2.imwrite(file_path, img)

def save_image_with_middle_points(img, detections, file_path):
    for detection in detections:
        if len(detection) < 6:
            continue

        x1, y1, x2, y2 = map(int, detection[:4])
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 9, (190, 200, 100), cv2.FILLED)
    
    cv2.imwrite(file_path, img)

def detect_strawberies(image_path, model_path):
    model = YOLO(model_path)  # Load the selected model
    img = cv2.imread(image_path)
    # Initialize counters for each class using list comprehension
    counts = {name: 0 for name in classnames}

    # Store detections
    detections = []

    results = model(img)

    for info in results:
        parameters = info.boxes
        for details in parameters:
            x1, y1, x2, y2 = details.xyxy[0]
            conf = details.conf[0]
            conf = int(conf * 100 + 0.5)
            class_detect = details.cls[0]
            class_detect = int(class_detect)

            if conf > 70:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                current_detection = [x1, y1, x2, y2, class_detect, conf]
                detections.append(current_detection)

                # Increment the count for this class
                if class_detect < len(classnames):
                    class_name = classnames[class_detect]
                    counts[class_name] += 1

    detections = np.array(detections)

    return detections, counts

def load_results():
    try:
        if os.path.exists('results.json'):
            with open('results.json', 'r') as file:
                data = file.read()
                if data.strip():  # Check if the file is not empty
                    return json.loads(data)
                else:
                    return []  # Return an empty list if the file is empty
        else:
            return []  # Return an empty list if the file does not exist
    except json.JSONDecodeError:
        return []  # Return an empty list if the file contains invalid JSON

def save_result(result):
    results = load_results()
    result['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results.append(result)
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

def update_paths_in_results():
    with open('results.json', 'r') as file:
        results = json.load(file)

    for result in results:
        result['image_path'] = result['image_path'].replace("static/", "").replace("\\", "/")
        result['bounding_boxes_img_path'] = result['bounding_boxes_img_path'].replace("static/", "").replace("\\", "/")
        result['middle_points_img_path'] = result['middle_points_img_path'].replace("static/", "").replace("\\", "/")

    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

def generate_unique_folder(upload_folder):
    folder_name = str(uuid.uuid4().hex[:8])  # Generate a unique folder name
    unique_folder = os.path.join(upload_folder, folder_name)
    os.makedirs(unique_folder, exist_ok=True)
    return unique_folder

def generate_unique_filename(upload_folder, filename):
    base_name, extension = os.path.splitext(filename)
    unique_name = f"{base_name}_{uuid.uuid4().hex[:6]}{extension}"
    return os.path.join(upload_folder, unique_name).replace("\\", "/")

def calculate_strawberries_count(detections, class_id):
    count = 0
    for detection in detections:
        if detection['class'] == class_id:
            count += 1
    return count

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the request contains a file
        if 'file' in request.files:
            file = request.files['file']
            selected_model = request.form['model']
            if file:
                filename = file.filename
                upload_folder = 'static/uploads'
                os.makedirs(upload_folder, exist_ok=True)

                # Generate a unique folder
                unique_folder = generate_unique_folder(upload_folder)
                
                # Define the path with the JPG extension regardless of the original format
                new_filename = os.path.splitext(filename)[0] + ".jpg"
                file_path = os.path.join(unique_folder, new_filename).replace("\\", "/")
                
                # Open the uploaded image file
                image = Image.open(file)
                
                # Resize the image to 1008x756
                resized_image = image.resize((1008, 756))
                
                # Convert and save the image in JPG format
                resized_image.save(file_path, format='JPEG')
                
                # Proceed with the detection using the resized and converted image
                detections, counts = detect_strawberies(file_path, model_paths[selected_model])
                bounding_boxes_img_path = os.path.join(unique_folder, 'bounding_boxes.jpg').replace("\\", "/")
                save_image_with_bounding_boxes(cv2.imread(file_path), detections, bounding_boxes_img_path)

                middle_points_img_path = os.path.join(unique_folder, 'middle_points.jpg').replace("\\", "/")
                _ = save_image_with_middle_points(cv2.imread(file_path), detections, middle_points_img_path)

                result = {
                    'filename': filename,
                    'model': selected_model,
                    'image_path': file_path,
                    'bounding_boxes_img_path': bounding_boxes_img_path,
                    'middle_points_img_path': middle_points_img_path,
                    'halfripe_strawberries_count': counts.get('halfripe', 0),
                    'ripe_strawberries_count': counts.get('ripe', 0),
                    'unripe_strawberries_count': counts.get('unripe', 0)
                }

                save_result(result)
                update_paths_in_results()

                return render_template('result.html', 
                                        filename=filename,
                                        model=selected_model,
                                        image_path=file_path,
                                        bounding_boxes_img_path=bounding_boxes_img_path,
                                        middle_points_img_path=middle_points_img_path,
                                        halfripe_strawberries_count=counts.get('halfripe', 0),
                                        ripe_strawberries_count=counts.get('ripe', 0),
                                        unripe_strawberries_count=counts.get('unripe', 0),
                        )
        # Handle case where file is a path for testing
        elif 'file' in request.form:
            image_name = request.form['file']
            selected_model = request.form['model']
            
            image_path = os.path.join('static/testdata', image_name)
            detections, counts = detect_strawberies(image_path, model_paths[selected_model])

            unique_folder = generate_unique_folder('static/uploads')
            bounding_boxes_img_path = os.path.join(unique_folder, 'bounding_boxes.jpg').replace("\\", "/")
            save_image_with_bounding_boxes(cv2.imread(image_path), detections, bounding_boxes_img_path)

            middle_points_img_path = os.path.join(unique_folder, 'middle_points.jpg').replace("\\", "/")
            _ = save_image_with_middle_points(cv2.imread(image_path), detections, middle_points_img_path)

            result = {
                'filename': image_name,
                'model': selected_model,
                'image_path': image_path,
                'bounding_boxes_img_path': bounding_boxes_img_path,
                'middle_points_img_path': middle_points_img_path,
                'halfripe_strawberries_count': counts.get('halfripe', 0),
                'ripe_strawberries_count': counts.get('ripe', 0),
                'unripe_strawberries_count': counts.get('unripe', 0)
            }

            save_result(result)
            update_paths_in_results()

            return render_template('result.html', 
                                    filename=image_name,
                                    model=selected_model,
                                    image_path=image_path,
                                    bounding_boxes_img_path=bounding_boxes_img_path,
                                    middle_points_img_path=middle_points_img_path,
                                    halfripe_strawberries_count=counts.get('halfripe', 0),
                                    ripe_strawberries_count=counts.get('ripe', 0),
                                    unripe_strawberries_count=counts.get('unripe', 0),
                    )

    test_images = os.listdir('static/testdata')
    test_images = [img for img in test_images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('index.html', test_images=test_images)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results')
def results():
    results = load_results()
    results.sort(key=lambda x: x['date'], reverse=True)
    return render_template('previous_results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)