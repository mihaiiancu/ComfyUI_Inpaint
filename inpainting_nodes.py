#Developed by m

#This node provides a simple interface to inpaint

import numpy as np
from .inpaint.mediapipe import *

class InpaintMediapipe:
    """
        
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        """
        Input Types

        """
        return {
            "required": {
				"inpaint_model": ("MODEL",),
                "image": ("IMAGE",),
                "inpaint_type": (["face", "hand", "body"],),
                "prompt": ("STRING", {"multiline": True}),
				"negative_prompt": ("STRING", {"multiline": True}), 
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), 
                "guidance_scale": ("FLOAT", {"default": 7, "min": 0, "max": 50, "step": 0.5}), 
                "confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "match_color": (["True", "False"],),
                "blur_factor": ("INT", {"default": 0, "min":0, "max":200, "step":1}),               
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("Inpainted","Annotated",)
    FUNCTION = "inpaint_mediapipe"
    CATEGORY = "image/postprocessing"

    def inpaint_mediapipe(self, inpaint_model, image, inpaint_type, prompt, negative_prompt, strength, guidance_scale, confidence, match_color, blur_factor):

        inpainted_image = image
        annotated_image = image
        # Choose a filter based on the 'mode' value
        match_col = True
        if match_color == "False":
            match_col = False
        
        if inpaint_type == "face":
            inpainted_image, annotated_image = inpaint_faces(inpaint_model, image, prompt, negative_prompt, strength, guidance_scale, confidence, match_col, blur_factor)
        elif inpaint_type == "hand":
            inpainted_image, annotated_image = inpaint_hands(inpaint_model, image, prompt, negative_prompt, strength, guidance_scale, confidence, match_col, blur_factor)
        elif inpaint_type == "body":
            inpainted_image, annotated_image = inpaint_people(inpaint_model, image, prompt, negative_prompt, strength, guidance_scale, confidence, match_col, blur_factor)
        else:
            print(f"Invalid inpaint_type option: {mode}. No changes applied.")
            return (inpainted_image,annotated_image)

        return (inpainted_image,annotated_image)

NODE_CLASS_MAPPINGS = {
    "InpaintMediapipe": InpaintMediapipe
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InpaintMediapipe": "Inpaint Mediapipe",
}
