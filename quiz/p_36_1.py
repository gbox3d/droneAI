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
tracker = None
tracking_active = False

# missile
missile_pos = []
missile_direction = []
missile_speed = 50 # pixel/sec
missile_is_activate = False

delta_tick = 0

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
    
initial_distance = None  # 최초 거리 기록
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
            
            init_tracker_bbox = tuple(map(int,current_box)) # to int
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame,init_tracker_bbox)
            tracking_active = True
            print(f"tracker OK {current_box}")
        else :
            print("SAM failed create BBox")
            current_box = None
        
        click_point = None
        sam_processing = None
    
    if tracking_active and tracker : 
        success,bbox_from_tracker = tracker.update(frame)
        if success:
            current_box = tuple(map(int,bbox_from_tracker)) # to int
        else :
            print("tracking missing")
            tracker = None
            tracking_active = False
            current_box = None
    
    # Convert the frame to RGB format 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #create pygame surface
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen_surface.blit(frame_surface, (0, 0))
    
    # draw bbox
    if current_box : 
        pygame.draw.rect(screen_surface,(0,255,0),current_box,2)
    
        if missile_is_activate :
            x,y,w,h = current_box
            target_center = [x+ w//2 , y+h//2]
            
            missile_direction = [target_center[0] - missile_pos[0] , 
                                 target_center[1] - missile_pos[1]]
            #normalize
            length = np.linalg.norm(missile_direction)
            
            if initial_distance is None:
                initial_distance = length  # 최초 거리 기록

            if length > 0:
                if length < 10:
                    missile_is_activate = False
                    initial_distance = None
                    continue
                
                missile_direction = [missile_direction[0] / length,
                                     missile_direction[1] / length]
                
            missile_pos[0] += missile_direction[0] * missile_speed * (delta_tick / 1000)
            missile_pos[1] += missile_direction[1] * missile_speed * (delta_tick / 1000)
            
            
            # 색상 변화 계산 (녹→노→빨)
            if initial_distance:
                ratio = max(0.0, min(length / initial_distance, 1.0))
                # 구간별 보간
                if ratio > 0.5:
                    t = (ratio - 0.5) * 2
                    color = (
                        int(255 * t),           # R
                        255,                    # G
                        0                       # B
                    )  # 녹→노
                else:
                    t = ratio * 2
                    color = (
                        255,                    # R
                        int(255 * t),           # G
                        0                       # B
                    )  # 노→빨
            else:
                color = (0, 255, 0)

            pygame.draw.circle(screen_surface, color, (int(missile_pos[0]), int(missile_pos[1])), 6)

    
    _text = font.render(f"SAM Tracking Demo - Click to select target and space to launch missile",True,(255,255,0))
    screen_surface.blit(_text,(10,height -30))

    pygame.display.flip()
    
    delta_tick = clock.tick(30) # Limit to 30 FPS
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bLoop = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                bLoop = False
            elif event.key == pygame.K_SPACE :
                if current_box and not missile_is_activate : 
                    missile_pos = [width // 2, height]
                    missile_direction = [0,0]
                    missile_is_activate = True
                    
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if not sam_processing:
                    click_point = list(event.pos)
                    current_box = None
                    

cap.release()
pygame.quit()
