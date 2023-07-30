import cv2
import mediapipe as mp
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

#inpaint_model = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to("cuda")

def inpaint_hands(inpaint_model, original_image, prompt, negative_prompt, strength, guidance_scale, confidence=0.5, match_color=False, blur_factor=0):

  original_copy = original_image.copy()
  
  # Detect hands
  mp_hands = mp.solutions.hands.Hands()
  results = mp_hands.process(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

  hand_bboxes = []
  for hand_landmarks in results.multi_hand_landmarks:
    if hand_landmarks.classification[0].score < confidence:
      continue
    brect = cv2.boundingRect(np.array([landmark.x, landmark.y] for landmark in hand_landmarks.landmark).reshape(-1, 2))
    hand_bboxes.append(brect)

  # Draw bboxes on copy
  for x,y,x2,y2 in hand_bboxes:
    cv2.rectangle(original_copy, (x,y), (x2, y2), (0,0,255), 2) 
	
  # Inpaint each hand
  original_image = impaint_image(inpaint_model, original_image, hand_bboxes, prompt, negative_prompt, strength, guidance_scale, match_color, blur_factor)

  return original_image, original_copy


def inpaint_faces(inpaint_model, original_image, prompt, negative_prompt, strength, guidance_scale, confidence=0.5, match_color=False, blur_factor=0):

  original_copy = original_image.copy()

  # Detect faces
  mp_face_mesh = mp.solutions.face_mesh
  face_mesh = mp_face_mesh.FaceMesh()
  results = face_mesh.process(original_image)

  # Get bboxes
  face_bboxes = []
  for face_landmarks in results.multi_face_landmarks:
    if face_landmarks.classification[0].score < confidence:
        continue
    pts = []
    for landmark in face_landmarks.landmark:
      pts.append([int(landmark.x * original_image.shape[1]), int(landmark.y * original_image.shape[0])])
    pts = np.array(pts)
    x,y,w,h = cv2.boundingRect(pts)
    face_bboxes.append([x,y,x+w,y+h])
    
  # Draw bboxes on copy
  for x,y,x2,y2 in face_bboxes:
    cv2.rectangle(original_copy, (x,y), (x2, y2), (0,0,255), 2) 

  # Inpaint each face
  original_image = impaint_image(inpaint_model, original_image, face_bboxes, prompt, negative_prompt, strength, guidance_scale, match_color, blur_factor)

  return original_image, original_copy

def inpaint_people(inpaint_model, original_image, prompt, negative_prompt, strength, guidance_scale, confidence=0.5, match_color=False, blur_factor=0):
  original_copy = original_image.copy()
  
  # Detect people
  mp_pose = mp.solutions.pose
  pose_model = mp_pose.Pose()
  results = pose_model.process(original_image)
  
  pose_bboxes = []
  
  for person in results.pose_landmarks:
    if person.classification[0].score < confidence:
      continue
    # Get bbox
    keypoints = np.array([[lmk.x, lmk.y] for lmk in person.landmark])  
    x, y, w, h = cv2.boundingRect(keypoints)
    pose_bboxes.append([x, y, x+w, y+h])
    
  # Draw bboxes on copy
  for x,y,x2,y2 in pose_bboxes:
    cv2.rectangle(original_copy, (x,y), (x2, y2), (0,0,255), 2) 

  original_image = impaint_image(inpaint_model, original_image, pose_bboxes, prompt, negative_prompt, strength, guidance_scale, match_color, blur_factor)
  
  return original_image, original_copy

def impaint_image(inpaint_model, image, bboxes, prompt, negative_prompt, strength, guidance_scale, match_color, blur_factor):

  # Inpaint each face
  for x,y,x2,y2 in bboxes:
  
	#Crop face
    image_crop = image[y:y2, x:x2]
	
	# Inpaint
    mask = Image.new("L", image_crop.size, 0)
	
	if blur_factor > 0:
      mask = mask.filter(ImageFilter.GaussianBlur(blur_factor))
	  
    inpainted = inpaint_model(prompt=prompt, negative_prompt=negative_prompt, image=image_crop, mask=mask, strength=strength,                    guidance_scale=guidance_scale).images[0]

	# Paste back
    image[y:y2, x:x2] = np.array(inpainted)
    
    # Color match
	if match_color:
      matched = color_match(inpainted, image)  
      image[y:y2, x:x2] = matched
      
  return image

def load_image(image_path):
    return cv2.imread(image_path)


# Color match function remains the same
def color_match(source, template):

  source = cv2.split(source)
  template = cv2.split(template)

  matched = []
  for i in range(3):
    
    s_hist = cv2.calcHist([source[i]], [0], None, [256], [0, 256])
    t_hist = cv2.calcHist([template[i]], [0], None, [256], [0, 256])

    matched.append(match_histogram(source[i], t_hist))

  return cv2.merge(matched)


def match_histogram(source, template):

  oldshape = source.shape
  source = source.ravel()
  template = template.ravel()

  s_values, s_idxs, s_counts = np.unique(source, return_inverse=True, return_counts=True)
  t_values, t_counts = np.unique(template, return_counts=True)
   
  s_quantiles = np.cumsum(s_counts).astype(np.float64)
  s_quantiles /= s_quantiles[-1]

  t_quantiles = np.cumsum(t_counts).astype(np.float64)
  t_quantiles /= t_quantiles[-1]
  
  interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

  return interp_t_values[s_idxs].reshape(oldshape)