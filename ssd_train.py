import torch
import torchvision
import torch.optim as optim
import cv2
import os

# Load SSD model
model = torchvision.models.detection.ssd300_vgg16(weights="DEFAULT")
model.train()

print("SSD model loaded")

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Image folder
image_folder = "coco128/images/train2017"
images = os.listdir(image_folder)[:10]

epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for img_name in images:

        img_path = os.path.join(image_folder, img_name)

        img = cv2.imread(img_path)

        img_tensor = torch.tensor(img).permute(2,0,1).float()/255

        # Dummy bounding box
        boxes = torch.tensor([[50,50,200,200]]).float()
        labels = torch.tensor([1])

        target = {"boxes":boxes,"labels":labels}

        optimizer.zero_grad()

        loss_dict = model([img_tensor],[target])

        loss = sum(loss for loss in loss_dict.values())

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch:",epoch+1,"Loss:",total_loss)
torch.save(model.state_dict(), "ssd_model.pth")
print("Model saved")