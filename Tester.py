import torch
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from ModelStructure import NeuralNet
import cv2

input_size = 784 #28*28
hidden_size = 100
num_classes = 10

test_file = torchvision.datasets.MNIST(root = 'data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_file, batch_size=100, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

File = 'model.pth'
model = NeuralNet(input_size, hidden_size, num_classes)
model.load_state_dict(torch.load(File, weights_only=True))
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

def MNIST_test():
    with torch.no_grad():
        n_correct = 0
        n_sample = 0
        for images,labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions= torch.max(outputs, 1)
            n_sample += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
        
        acc = 100.0 * n_correct / n_sample

        print(f'accuracy = {acc:.4f}')

def Personal_test():
    with torch.no_grad():
        image = input('Enter file PATH: ')
        img = cv2.imread(image,cv2.IMREAD_GRAYSCALE) #Read the image as a grayscale
        img = img.astype(np.float32)  #Format image as 
        img = cv2.resize(img, (28,28))  #Resize the data to the MNIST dimensions
        imgTorch = torch.from_numpy(img)
        imgTorch = imgTorch.reshape(-1, 28*28).to(device) #Get the image in the form of an array
        output = model(imgTorch)
        _, predictions= torch.max(output, 1)
        prediction = predictions.__getitem__(0)

        print(f"System predicted: {prediction} for {image}")

option = 0

while option !=3:
    option = int(input(f"Enter:\n\t1 to test model using MNIST dataset.\n\t2 To test using a personalized image.\n\t3 to quit program.\n"))
    if(option == 1):
        MNIST_test()
    elif(option ==2):
        Personal_test()
    elif(option != 3):
        print("You have entered an invalid option.")