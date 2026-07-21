import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]
loss = [171.51, 70.52, 50.79, 39.54, 32.80]

plt.plot(epochs, loss, marker='o')

plt.title("SSD Training Results")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")

plt.show()