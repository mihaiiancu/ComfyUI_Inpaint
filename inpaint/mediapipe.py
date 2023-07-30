import cv2
import mediapipe as mp
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import PIL
import torch

#inpaint_model = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to("cuda")

def inpaint_hands(inpaint_model, tensor_image, prompt, negative_prompt, strength, guidance_scale, confidence=0.5, match_color=False, blur_factor=0):
    print(type(tensor_image))
    original_image = pil_to_cv2(tensor_to_pil(tensor_image))
    original_copy = original_image.copy()
  
    # Detect hands
    mp_hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=confidence, min_tracking_confidence=confidence)
    results = mp_hands.process(original_image)

    multi_hand_landmarks = results.multi_hand_landmarks
    if multi_hand_landmarks:
        hand_bboxes = []
        for hand_landmarks in multi_hand_landmarks:
            #if hand_landmarks.classification[0].score < confidence:
            #  continue
            pts = []
            for landmark in hand_landmarks.landmark:
                pts.append([int(landmark.x * original_image.shape[1]), int(landmark.y * original_image.shape[0])])
            pts = np.array(pts)
            x,y,w,h = cv2.boundingRect(pts)
            hand_bboxes.append([x,y,x+w,y+h])

        # Draw bboxes on copy
        for x,y,x2,y2 in hand_bboxes:
            cv2.rectangle(original_copy, (x,y), (x2, y2), (0,0,255), 5) 
            
        # Inpaint each hand
        original_image = impaint_image(inpaint_model, original_image, hand_bboxes, prompt, negative_prompt, strength, guidance_scale, match_color, blur_factor)
    else:
        print("No hands detected")
        
    return pil_to_tensor(cv2_to_pil(original_image)), pil_to_tensor(cv2_to_pil(original_copy))


def inpaint_faces(inpaint_model, tensor_image, prompt, negative_prompt, strength, guidance_scale, confidence=0.5, match_color=False, blur_factor=0):
    print(type(tensor_image))
    print(tensor_image.shape)
    
    original_image = pil_to_cv2(tensor_to_pil(tensor_image))
    original_copy = original_image.copy()

    # Detect faces
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, refine_landmarks=True, min_detection_confidence=confidence)
    results = face_mesh.process(original_image)
    #print(results)
    face_landmarks = results.multi_face_landmarks
    # Get bboxes
    
    if face_landmarks:
        face_bboxes = []
        num_faces = len(face_landmarks)
        print(num_faces)
        for face_landmark in face_landmarks:
            #print("-------<>",face_landmarks)
            #if face_landmarks.classification[0].score < confidence:
            #    continue
            pts = []
            for landmark in face_landmark.landmark:
                pts.append([int(landmark.x * original_image.shape[1]), int(landmark.y * original_image.shape[0])])
            pts = np.array(pts)
            x,y,w,h = cv2.boundingRect(pts)
            face_bboxes.append([x,y,x+w,y+h])

        # Draw bboxes on copy
        for x,y,x2,y2 in face_bboxes:
            cv2.rectangle(original_copy, (x,y), (x2, y2), (0,0,255), 5) 

        # Inpaint each face
        original_inpainted_image = impaint_image(inpaint_model, original_image, face_bboxes, prompt, negative_prompt, strength, guidance_scale, match_color, blur_factor)

    else:
        print("No faces detected")
        
    return pil_to_tensor(cv2_to_pil(original_image)), pil_to_tensor(cv2_to_pil(original_copy))

def inpaint_people(inpaint_model, tensor_image, prompt, negative_prompt, strength, guidance_scale, confidence=0.5, match_color=False, blur_factor=0):
    original_image = pil_to_cv2(tensor_to_pil(tensor_image))
    original_copy = original_image.copy()
  
    # Detect people
    mp_pose = mp.solutions.pose
    pose_model = mp_pose.Pose()
    results = pose_model.process(original_image)
  
    pose_landmarks = results.pose_landmarks
    """
    # Check if any poses detected 
    if pose_landmarks:
        pose_bboxes = []
        if type(pose_landmarks) is list: 
            num_poses = len(pose_landmarks)
            # Add bounding box for each detected pose
            for i in range(num_poses):
                pose_landmark = pose_landmarks[i]
                #if person.classification[0].score < confidence:
                #    continue
                # Get bbox
                pts = []
                for landmark in pose_landmark.landmark:
                    pts.append([int(landmark.x * original_image.shape[1]), int(landmark.y * original_image.shape[0])])
                pts = np.array(pts)
                x,y,w,h = cv2.boundingRect(pts)
                pose_bboxes.append([x,y,x+w,y+h])
        else:
            #if person.classification[0].score < confidence:
            #    continue
            # Get bbox
            pts = []
            for landmark in pose_landmarks.landmark:
                pts.append([int(landmark.x * original_image.shape[1]), int(landmark.y * original_image.shape[0])])
            pts = np.array(pts)
            x,y,w,h = cv2.boundingRect(pts)
            pose_bboxes.append([x,y,x+w,y+h])            

        # Draw bboxes on copy
        for x,y,x2,y2 in pose_bboxes:
            cv2.rectangle(original_copy, (x,y), (x2, y2), (0,0,255), 2) 

        original_image = impaint_image(inpaint_model, original_image, pose_bboxes, prompt, negative_prompt, strength, guidance_scale, match_color, blur_factor)
    else:
      print("No poses detected")
    """
    
    if pose_landmarks:
        pose_bboxes = []
        landmarks = results.pose_landmarks.landmark

        # Get visible landmarks 
        pts = []
        for landmark in landmarks:
            if landmark.visibility > 0.1:
                pts.append([int(landmark.x * original_image.shape[1]), int(landmark.y * original_image.shape[0])])
                
        pts = np.array(pts)
        x,y,w,h = cv2.boundingRect(pts)
        pose_bboxes.append([x,y,x+w,y+h])
        
        # Draw bboxes on copy
        for x,y,x2,y2 in pose_bboxes:
            cv2.rectangle(original_copy, (x,y), (x2, y2), (0,0,255), 2) 

        original_image = impaint_image(inpaint_model, original_image, pose_bboxes, prompt, negative_prompt, strength, guidance_scale, match_color, blur_factor)
        
    else:
        print("No poses detected")
    
  
    return pil_to_tensor(cv2_to_pil(original_image)), pil_to_tensor(cv2_to_pil(original_copy))

def impaint_image(inpaint_model, image, bboxes, prompt, negative_prompt, strength, guidance_scale, match_color, blur_factor):

    print(type(inpaint_model.model))
    # Inpaint each face
    for x,y,x2,y2 in bboxes:
  
        #Crop face
        image_crop = image[y:y2, x:x2]
        #print("-----<>",image_crop)
        # Inpaint
        mask = Image.new("L", cv2_to_pil(image_crop).size, 0)
        
        if blur_factor > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur_factor))
          
        #inpainted = inpaint_model(prompt=prompt, negative_prompt=negative_prompt, image=image_crop, mask=mask, strength=strength,                    guidance_scale=guidance_scale).images[0]

        # Paste back
        #image[y:y2, x:x2] = np.array(inpainted)
        
        # Color match
        #if match_color:
        #    matched = color_match(inpainted, image)  
        #    image[y:y2, x:x2] = matched
      
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
    
def tensor_to_pil(tensor_image):
    """
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)
    """
    #image = 255. * tensor_image[0].cpu().numpy()
    #return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    return Image.fromarray(np.clip(255. * tensor_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
def pil_to_tensor(original_image):
    original_image_np = np.array(original_image).astype(np.float32) / 255.0
    return torch.from_numpy(original_image_np).unsqueeze(0)

def cv2_to_pil(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)