import torch
import torch.nn.functional
from torchvision import datasets
import torch.utils.data
from MyAlexNet import MyAlexNet1, MyAlexNet2, MyAlexNet3, MyAlexNet4
from Cifar10Dataset import Cifar10Dataset
from torch.utils.tensorboard import SummaryWriter

# 定义全局变量
n_epochs = 100  # epoch 的数目
batch_size = 100  # 决定每次读取多少图片
learn_rate = 3e-5 # 学习率

train_dataset = Cifar10Dataset('./cifar-10-batches-py/', train=True)
test_dataset = Cifar10Dataset('./cifar-10-batches-py/', train=False)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


def train(model, log_name):
    # 定义损失函数和优化器
    writer = SummaryWriter(f'./log/' + log_name + '/')
    lossfunc = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)
    # 开始训练
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 得到预测值
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('\tEpoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        accuracy = test(model)
        writer.add_scalar('train/Loss', train_loss, epoch + 1)
        writer.add_scalar('test/Accuracy', accuracy, epoch + 1)
        writer.flush()
    writer.close()


# 在数据集上测试神经网络
def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集中不需要反向传播
        for data in test_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)   # 概率最大的就是输出的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 有几个相同的
    print('\ttest Accuracy: %.4f %%' % (
            100 * correct / total))
    return 100.0 * correct / total

if __name__ == '__main__':
    print('MyAlexNet1:')
    AlexNet1 = MyAlexNet1().cuda()
    train(AlexNet1, 'AlexNet1')
    torch.save(AlexNet1, 'AlexNet1.ckpt')
    
    print('MyAlexNet2:')
    AlexNet2 = MyAlexNet2().cuda()
    train(AlexNet2, 'AlexNet2')
    torch.save(AlexNet2, 'AlexNet2.ckpt')
    
    print('MyAlexNet3:')
    AlexNet3 = MyAlexNet3().cuda()
    train(AlexNet3, 'AlexNet3')
    torch.save(AlexNet3, 'AlexNet3.ckpt')
    
    print('MyAlexNet4:')
    AlexNet4 = MyAlexNet4().cuda()
    train(AlexNet4, 'AlexNet4')
    torch.save(AlexNet4, 'AlexNet4.ckpt')