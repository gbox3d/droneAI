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

#load sample image
frame = cv2.imread("chapter02/bus.jpg")
if frame is None:
    print("Error: Could not read the image.")
    sys.exit(1)
print(f"load image OK , Image shape: {frame.shape}")

# resize the frame to fit the screen
orig_h, orig_w = frame.shape[:2]

# target dimensions
target_w = width
target_h = height

#calculate scaling ratio 
ratio_w = target_w / orig_w
ratio_h = target_h / orig_h

scale_ratio = min(ratio_w, ratio_h)

# resize the frame
new_w = int(orig_w * scale_ratio)
new_h = int(orig_h * scale_ratio)
frame = cv2.resize(frame, (new_w, new_h),interpolation=cv2.INTER_AREA)

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))

screen_surface.blit(frame_surface, (0, 0))

# Run inference on the image
results = model(frame,conf=0.5)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        conf = int(box.conf[0] * 100)
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf}%"
        pygame.draw.rect(screen_surface, (0, 255, 0), (x1, y1, x2 - x1, y2 - y1), 2)
        text_surface = font.render(label, True, (255, 255, 255))
        screen_surface.blit(text_surface, (x1, y1 - 20))

pygame.display.flip()

bLoop = True
while bLoop:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            bLoop = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                bLoop = False

pygame.quit()
print("Exiting the program.")




