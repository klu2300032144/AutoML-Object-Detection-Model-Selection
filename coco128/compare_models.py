# Model comparison results (from your experiments)

models = {
    "YOLOv8n": 0.59,
    "YOLOv8s": 0.70,
    "SSD": 0.40
}

print("MODEL PERFORMANCE COMPARISON\n")

best_model = max(models, key=models.get)

for model, score in models.items():
    print(f"{model} mAP50 Accuracy: {score}")

print("\nBest Model:", best_model)
print("Highest Accuracy:", models[best_model])