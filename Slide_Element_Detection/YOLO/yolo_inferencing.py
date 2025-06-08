from ultralytics import YOLO
import os
from tqdm import tqdm
import cv2
import pandas as pd 

val_images_path = 'path_to_images_folder'
inference_out_folder = 'path_to_output_folder'
model_path = 'trained_model_path'
model = YOLO(model_path)

# Ensure the output folder exists
os.makedirs(inference_out_folder, exist_ok=True)
df=pd.DataFrame(columns=["image","classes"])

l_1=[]
l_2=[]
for img in tqdm(os.listdir(val_images_path)):
    img_path = os.path.join(val_images_path, img)
    l_1.append(img)
    image = cv2.imread(img_path)
    
    pred = model(img_path)
    class_l=''
    
    for result in pred:
        boxes = result.boxes  # Boxes object for bounding box outputs
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Convert to numpy
            
            # Extract class name and confidence score
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            confidence = float(box.conf[0].cpu().numpy())  # Convert tensor to float
            
            class_l += f"{class_name} ({confidence:.2f}), "
            
            # Draw the bounding box on the image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue color in BGR
            
            # Put class name and confidence score on the bounding box
            cv2.putText(image, f"{class_name} ({confidence:.2f})", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    l_2.append(class_l.strip(', '))
    
    # Save the image with bounding boxes and class names
    output_path = os.path.join(inference_out_folder, img)
    cv2.imwrite(output_path, image)
    
df['image'] = l_1
df['classes'] = l_2
df.to_csv('path_to_csv', index=False)
