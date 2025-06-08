from ultralytics import YOLO
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'



model_path= 'path_to_model'

data_yaml='path_to_configfile'
output_dir= 'path_to_outputfolder'
epochs= 10
img_size= 640
batch_size = 16
# Initialize the model
model = YOLO(model_path)  # Load the model configuration and weights


# print(f"Starting training for epoch {epoch+1}/{epochs}")
# Train the model for one epoch
train_results = model.train(
    data=data_yaml,        # Path to data yaml file
    epochs=epochs,              # Training for one epoch at a time
    imgsz=(360,640),        # Input image size
    batch=batch_size,      # Batch size
    project=output_dir,    # Directory to save training outputs
    name=f'exp_epoch_{epochs}',  
    device='0,1,2,4',# Experiment name, unique per epoch
    exist_ok=True,
    augment=False,
    hsv_h=0,
    hsv_s=0,
    hsv_v=0, degrees=0.0, translate=0, scale=0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0, mosaic=0, mixup=0.0)#Overwrite existing data if necessary
    

print(f"Training for epoch {epochs} completed.")

