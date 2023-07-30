import cv2
import torch
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torchvision.transforms as T

# Download and load UltraFace model
model = YOLO("ultralytics/yolov8n-face")
model.eval()

inpaint_model = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to("cuda")

def inpaint(model, image_path, prompt, negative_prompt, strength, guidance_scale, confidence=0.5, match_color=False, blur_factor=0):

  original_image = cv2.imread(image_path)
  original_copy = original_image.copy()

  # Detect faces
  results = model(original_image)
  #faces = results.boxes
  faces = [box for *box, conf, cls in results.boxes if conf >= conf_threshold]
  
  # Draw bboxes on copy
  for *box, conf, cls in faces:
    x1, y1, x2, y2 = box
    cv2.rectangle(original_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

  # Inpaint each face
  for *box, conf, cls in faces:
    x1, y1, x2, y2 = box
    
    face_crop = original_image[int(y1):int(y2), int(x1):int(x2)]

    # Create mask
    mask = Image.new("L", face_crop.size, 0)
    mask = T.ToTensor()(mask).unsqueeze(0).to(inpaint_model.device)
    
    # Inpaint face
    inpainted = inpaint_model(prompt=prompt, negative_prompt=negative_prompt, image=face_crop, mask=mask, strength=strength,                    guidance_scale=guidance_scale).images[0]

    # Color correct
    matched = color_match(inpainted, original_image)

    original_image[int(y1):int(y2), int(x1):int(x2)] = matched

  return original_image, original_copy


# Color match function