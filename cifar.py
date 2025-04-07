import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

trainset=torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True)

testset=torchvision.datasets.CIFAR10(root='./data', train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64*6*6,600)
        self.fc2 = nn.Linear(600,120)
        self.fc3 = nn.Linear(120,10)

    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.view(x.size(0), -1)  # flatten
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

model=CNN().to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

epochs=10
for epoch in range(epochs):
    running_loss=0.0
    for i, (images, labels) in enumerate(trainloader):
        images, labels=images.to(device),labels.to(device)

        outputs=model(images)
        loss=criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
    print(f"Epoch[{epoch+1}/{epochs}],Loss: {running_loss/len(trainloader):.4f}")

print("Finished Training")

correct=0
total=0
with torch.no_grad():
    for images,labels in testloader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _, predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

print(f"Test Accuracy:{100*correct/total:.2f}%")
