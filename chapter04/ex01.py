import cv2
import pygame
import numpy as np
import sys

from ultralytics import SAM

#model loading
model = SAM("sam2_s.pt")

width, height = 640, 480
# Initialize pygame
pygame.init()

screen_surface = pygame.display.set_mode((width, height))

# camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# pygame clock ,font
clock = pygame.time.Clock()
font = pygame.font.SysFont(None,24)

click_point = None
mask_bool = None

bLoop = True
while bLoop:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    #color conversion
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen_surface.blit(frame_surface, (0, 0))
    
    if click_point is not None:
        #pygame.draw.circle(screen_surface, (255, 0, 0), click_point, 5)
        results = model(frame,points=[click_point],labels=[1])
        
        if results and results[0].masks is not None and len(results[0].masks.data) > 0:
            mask_bool = results[0].masks.data[0].cpu().numpy().astype(bool)
        else :
            mask_bool = None
            print("No mask found in results.")
        click_point = None
        
    if mask_bool is not None:
        # Create a mask surface
        mask_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        mask_surface.fill((0, 0, 0, 0)) # Transparent background
        
        mask_bool_swapped = mask_bool.swapaxes(0, 1)
        
        # Convert the boolean mask to a color mask
        pixels_rgb = pygame.surfarray.pixels3d(mask_surface)
        pixels_alpha = pygame.surfarray.pixels_alpha(mask_surface)
        
        pixels_rgb[mask_bool_swapped] = [0,255,0]  # Green color for the mask
        pixels_alpha[mask_bool_swapped] = 128
        
        del pixels_rgb
        del pixels_alpha
        
        screen_surface.blit(mask_surface, (0, 0))
        
        
    #SAM inference
    #results = model(frame,points)
    
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bLoop = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                click_point = event.pos
                print(f"Clicked at: {click_point}")

    # Cap the frame rate
    clock.tick(30)
 