import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class MLP(nn.Module):
    def __init__(self,n_in=784,n_mid_units=100,n_out=10):
        super(MLP,self).__init__()

        self.n_in=n_in

        #Wx+b
        self.l1=nn.Linear(n_in,n_mid_units)
        self.l2=nn.Linear(n_mid_units,n_mid_units)
        self.l3=nn.Linear(n_mid_units,n_out)

    def forward(self,x):
        x2=x.view(-1,self.n_in)
        h1=F.relu(self.l1(x2))
        h2=F.relu(self.l2(h1))
        h3=self.l3(h2)

        return F.log_softmax(h3,dim=1)

if __name__=="__main__":
    trainset=torchvision.datasets.MNIST(
        ".",train=True,download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))
        ]))
    testset=torchvision.datasets.MNIST(
        ".",train=False,download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))
        ]))
    
    batch_size=128
    train_loader=torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=False)

    net=MLP()
    optimizer=optim.SGD(net.parameters(),lr=0.01)

    gpu_id=-1
    device=torch.device("cuda:{}".format(gpu_id) if gpu_id>=0 else "cpu")
    net=net.to(device)

    epochs=10
    log_interval=100

    #Training
    for epoch in range(1,epochs+1):
        net.train()

        for batch_idx,(data,target) in enumerate(train_loader):
            data,target=data.to(device),target.to(device)

            #勾配の初期化
            optimizer.zero_grad()
            #順伝播を計算
            output=net(data)
            #損失を計算
            loss=F.nll_loss(output,target)
            #逆伝播で勾配を計算
            loss.backward()
            #パラメータを更新
            optimizer.step()

            if batch_idx%log_interval==0:
                log_str="Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,batch_idx*len(data),len(train_loader.dataset),
                    100.0*batch_idx/len(train_loader),loss.item())
                print(log_str)
    
    #Evaluation
    net.eval()

    test_loss=0
    correct=0

    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=net(data)
            test_loss+=F.nll_loss(output,target,reduction="sum").item()
            pred=output.argmax(dim=1,keepdim=True)
            correct+=pred.eq(target.view_as(pred)).sum().item()

    test_loss/=len(test_loader.dataset)

    result_str="テストデータにおける平均損失: {:.4f}, 精度: {}/{} ({:.0f}%)".format(
        test_loss,correct,len(test_loader.dataset),
        100.0*correct/len(test_loader.dataset))
    print(result_str)

    #モデルを保存しておく
    torch.save(net.state_dict(),"pytorch_model.bin")
