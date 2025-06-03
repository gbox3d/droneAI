import cv2
import pygame

import sys

from ultralytics import YOLO,checks

checks()

#yolo11n load model
model = YOLO("yolo11n.pt")

width, height = 640, 480

# Initialize pygame
pygame.init()
screen_surface = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

#open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

bLoop = True
while bLoop:
    
    ret , frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen_surface.blit(frame_surface, (0, 0))
    
    # predict with YOLO model
    results = model.track(frame,
                          conf=0.7,
                          stream=True,
                          persist=True, 
                          verbose=False)
    
    # draw results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = int(box.conf[0] * 100)
            cls = int(box.cls[0])
            id = int(box.id) if box.id is not None else -1
            label = f"{model.names[cls]}: {conf:.2f} , ID: {id}"
            
            # Draw rectangle and label
            pygame.draw.rect(screen_surface, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
            text_surface = font.render(label, True, (255,0, 0))
            screen_surface.blit(text_surface, (x1, y1 - 20))
    
    
    #FPS display
    fps = clock.get_fps()
    fps_text = font.render(f"FPS: {fps:.2f}", True, (0, 0, 255))
    screen_surface.blit(fps_text, (10, 10))
    
    pygame.display.flip()
    clock.tick(30)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bLoop = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                bLoop = False

# Close the webcam
cap.release()
pygame.quit()

print("Exit successfully.")

