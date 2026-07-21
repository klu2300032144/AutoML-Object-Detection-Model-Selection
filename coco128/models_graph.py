import matplotlib.pyplot as plt

# Model names
models = ['YOLOv8n', 'YOLOv8s', 'SSD']

# Accuracy (mAP50) values from your results
accuracy = [0.59, 0.70, 0.40]

plt.figure()

plt.bar(models, accuracy)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("mAP50 Accuracy")

plt.show()