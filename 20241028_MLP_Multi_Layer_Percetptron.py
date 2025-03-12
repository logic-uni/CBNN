import torch

import torchvision.datasets

from torch import nn

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter



#准备数据

train_data = torchvision.datasets.FashionMNIST("./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)

test_data = torchvision.datasets.FashionMNIST("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)



#length 长度

train_data_size = len(train_data)

test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))

print("测试数据集的长度为：{}".format(test_data_size))



#利用dataloader来加载数据集

train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True)

test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)



#创建模型

class Net(nn.Module):

  def __init__(self):

    super(Net, self).__init__()

    self.model = nn.Sequential(

      nn.Flatten(),

      nn.Linear(784,256),

      nn.ReLU(),

      nn.Linear(256,10)

    )

  def forward(self,x):

    x = self.model(x)

    return x



net = Net()



#损失函数

loss_fun = nn.CrossEntropyLoss()



#优化器

lr = 0.01

optimizer = torch.optim.SGD(net.parameters(),lr=lr)



#设置训练网络的一些参数

#训练次数

total_train_step = 0

#测试次数

total_test_step = 0

#训练轮次

epoch = 10



#tensorboard

writer = SummaryWriter("./logs")



for i in range(epoch):

  print("-----第{}轮训练开始----".format(i+1))



  #开始训练

  net.train()

  for data in train_dataloader:

    imgs,targets = data

    outputs = net(imgs)

    loss = loss_fun(outputs,targets)

    #优化器优化模型

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    total_train_step += 1

    if total_train_step % 100 == 0:

      print("训练次数：{}，loss:{}".format(total_train_step, loss.item()))#加个item(),输出时为数字，不会有个tensor

      writer.add_scalar("train_loss",loss.item(),total_train_step)



  #测试步骤

  net.eval()

  total_test_loss = 0

  total_accuracy = 0

  with torch.no_grad():

    for data in test_dataloader:

      imgs,targets = data

      outputs = net(imgs)

      loss = loss_fun(outputs,targets)

      total_test_loss = total_test_loss + loss.item()

      accuracy = (outputs.argmax(1) == targets).sum()

      total_accuracy = total_accuracy + accuracy



  print(total_accuracy)

  print(test_data_size)

  print("整体测试集上的loss：{}".format(total_test_loss))

  print("整体测试集上的正确率：{}".format(total_accuracy.item() / test_data_size))

  writer.add_scalar("test_loss", total_test_loss, total_test_step)

  writer.add_scalar("test_accuracy", total_accuracy.item() / test_data_size, total_test_step)

  total_test_step += 1



writer.close()
