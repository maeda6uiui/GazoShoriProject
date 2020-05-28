import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

if __name__=="__main__":
    print("データローダの準備中......")
    transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    batch_size=1024
    trainset=torchvision.datasets.CIFAR10(
        root=".",train=True,download=True,transform=transform)
    trainloader=torch.utils.data.DataLoader(
        trainset,batch_size=batch_size,shuffle=True,num_workers=2)
    testset=torchvision.datasets.CIFAR10(
        root=".",train=False,download=True,transform=transform)
    testloader=torch.utils.data.DataLoader(
        testset,batch_size=batch_size,shuffle=False,num_workers=2)

    classes=("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")

    print("モデルの準備中......")
    net=Net()
    net.cuda()

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

    #Training
    print("訓練開始")

    epoch_num=50
    net.train()

    for epoch in range(epoch_num):
        print("========== Epoch {} / {} ==========".format(epoch+1,epoch_num))

        for i,data in enumerate(trainloader):
            inputs,labels=data

            inputs=inputs.cuda()
            labels=labels.cuda()

            optimizer.zero_grad()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

        print("損失: {:.6f}".format(loss.item()))

    torch.save(net.state_dict(),"pytorch_model.bin")
    print("訓練終了")

    #Test
    print("テスト開始")

    correct=0
    total=0
    net.eval()

    with torch.no_grad():
        for data in testloader:
            inputs,labels=data
            inputs=inputs.cuda()
            labels=labels.cuda()

            outputs=net(inputs)

            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()

    print("テスト終了")
    print("正解率: {:.6f}%".format(100.0*correct/total))
