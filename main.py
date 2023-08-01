import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'Input image')
ap.add_argument('-cl', '--classes', required=True,
                help = 'Text file containing class names')
args = ap.parse_args()

faster = True
threshold = 0.6

model = attempt_load('yolov7.pt', map_location='cpu')

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Pre processing the image
def process(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame)
    frame = frame / 255
    frame = frame.float()
    frame = frame.permute(2, 0, 1)
    frame = frame.unsqueeze(0)
    return frame

img = cv2.imread(args.image)
print(img.shape)
img = cv2.resize(img , (640, 480))

img_new = process(img)

with torch.inference_mode():
    preds = model(img_new)[0]
    preds = non_max_suppression(preds)

frame = img
for j in preds :
    j = j.clone()
    for (x1, y1, x2, y2, conf, cls) in j :
        if conf > threshold :
            cv2.putText(frame, classes[int(cls)], (int(x1), int(y1) - 10),  cv2.FONT_HERSHEY_DUPLEX, 0.9, colors[int(cls)], 2)
            frame = cv2.rectangle(frame,(int(x1), int(y1)), (int(x2), int(y2)) ,color = colors[int(cls)], thickness = 2)

cv2.imshow("Result", frame)
cv2.waitKey()
cv2.destroyAllWindows()

