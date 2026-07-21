import torch
import torchvision
import cv2
from torchvision.transforms import functional as F

# Load SSD model
model = torchvision.models.detection.ssd300_vgg16(weights=None)
model.load_state_dict(torch.load("ssd_model.pth"))
model.eval()

print("Model loaded")

# Load image
img = cv2.imread("coco128/images/train2017/000000000009.jpg")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = F.to_tensor(img_rgb)

with torch.no_grad():
    prediction = model([img_tensor])[0]

boxes = prediction['boxes']
scores = prediction['scores']

for box, score in zip(boxes, scores):

    if score > 0.1:

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imshow("SSD Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()