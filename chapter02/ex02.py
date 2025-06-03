import cv2
import pygame

import sys

from ultralytics import YOLO,checks

checks()

width, height = 640, 480

# Initialize pygame
pygame.init()
screen_surface = pygame.display.set_mode((width, height))

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)
print("Webcam opened successfully.")

bLoop = True
while bLoop:
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break
    
    # convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    screen_surface.blit(frame_surface, (0, 0))
    
    results = model(frame, conf=0.7)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = int(box.conf[0] * 100)
            cls = int(box.cls[0])
            
            # Draw the bounding box
            pygame.draw.rect(screen_surface, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
            
            # Prepare the label
            label = f"Class: {cls}, Conf: { model.names[cls]}%"
            text_surface = font.render(label, True, (255, 0, 0))
            screen_surface.blit(text_surface, (x1, y1 - 20))
            
    # fps calculation
    fps = clock.get_fps()
    fps_text = f"FPS: {fps:.2f}"
    fps_surface = font.render(fps_text, True, (0, 0, 255))
    screen_surface.blit(fps_surface, (10, 10))
    
    pygame.display.flip()
    clock.tick(30)  # prevent high CPU usage
    
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bLoop = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                bLoop = False

pygame.quit()
print("Exiting the program.")




