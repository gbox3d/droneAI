import cv2
import pygame

import sys
import numpy as np
from ultralytics import SAM

width, height = 640, 480

MODEL_NAME = "sam2_s.pt"

# Initialize the SAM model
try:
    model = SAM(MODEL_NAME)
    print(f"Model {MODEL_NAME} loaded successfully.")
except Exception as e:
    print(f"Error loading model {MODEL_NAME}: {e}")
    sys.exit(1)

#opencv camera capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Initialize pygame
pygame.init()
screen_surface = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# status variable
click_point = None
current_box = None
sam_processing = False


# mask -> bbox
def get_bbox_from_mask(mask_bool) : 
    if mask_bool is None or not np.any(mask_bool) :
        return None
    
    contours,_ = cv2.findContours(mask_bool.astype(np.uint8)*255,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if contours : 
        largest_contours = max(contours,key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(largest_contours)
        return (x,y,w,h)

bLoop = True
while bLoop:
    ret , frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video capture.")
        break
    
    if click_point :
        sam_processing = True
        print(f"SAM prompt : {click_point}")
        
        # SAM processing...
        results = model(frame,points=[click_point],labels=[1])
        initial_bbox_from_sam = None
        
        if results and results[0].masks : 
            mask_data = results[0].masks.data[0].cpu().numpy()
            mask_bool = mask_data.astype(bool)
            initial_bbox_from_sam = get_bbox_from_mask(mask_bool)
        
        if initial_bbox_from_sam :
            current_box = initial_bbox_from_sam
            print(f"SAM Bound Box Created Ok")
        
        
        click_point = None
        sam_processing = None
    
    
    # Convert the frame to RGB format 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #create pygame surface
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen_surface.blit(frame_surface, (0, 0))
    
    # draw bbox
    if current_box : 
        pygame.draw.rect(screen_surface,(0,255,0),current_box,2)
    
    pygame.display.flip()
    
    clock.tick(30) # Limit to 30 FPS
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bLoop = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                bLoop = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if not sam_processing:
                    click_point = list(event.pos)
                    current_box = None
                    

cap.release()
pygame.quit()



