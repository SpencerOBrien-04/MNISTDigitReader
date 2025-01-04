import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
from ModelStructure import NeuralNet
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyper parameter
input_size = 784 #28*28
hidden_size = 100
num_classes = 10
num_epochs = 40
batch_size = 100
learning_rate = 0.001

train_file = torchvision.datasets.MNIST(root = 'data', train=True, transform=transforms.ToTensor(), download=True)
test_file = torchvision.datasets.MNIST(root = 'data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_file, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_file, batch_size=batch_size, shuffle=False)

model = NeuralNet(input_size, hidden_size, num_classes)
model.to("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i + 1) % 100 == 0:
            print(f'Epoch: {epoch+1}\tStep:{i+1}\tLoss:{loss.item():.4f}')

torch.save(model.state_dict(),'model.pth')