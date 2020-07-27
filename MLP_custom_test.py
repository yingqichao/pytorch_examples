from __future__ import print_function
import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from mydataset.TwitterDataset import TwitterDataset
import setting

# Settings
lamb = 0.01 # 0.001 maybe is too small,loss is usually 0.006
input_dim = setting.input_layer_num # 32
output_dim = setting.output_layer_num # 16
hidden1_dim, hidden2_dim, feature_dim = 256, 256, 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # stage 1:find features via MLP, Input dimen: 32*1
        # self.linear1 = nn.Linear(32, 256)
        # self.linear2 = nn.Linear(256, 512)
        # self.linear0 = nn.Linear(512, 512)
        # self.linear3 = nn.Linear(512, 256)
        # self.feature_layer = nn.Linear(256,64)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),#layer 1
            # nn.Dropout(0.25),
            nn.LeakyReLU(),

            nn.Linear(hidden1_dim, hidden1_dim),#layer 2
            # nn.Dropout(0.25),
            nn.LeakyReLU(),

            # nn.Linear(hidden1_dim, hidden1_dim),#layer 3
            # nn.Dropout(0.25),
            # nn.LeakyReLU(),
            # nn.Linear(hidden1_dim, hidden1_dim),#layer 4
            # # nn.Dropout(0.25),
            # nn.LeakyReLU(),
            # nn.Linear(hidden1_dim, hidden1_dim),#layer 5
            # nn.Dropout(0.25),
            # nn.LeakyReLU(),
            # nn.Linear(hidden1_dim, hidden1_dim),#layer 6
            # # nn.Dropout(0.25),
            # nn.ReLU(),
            nn.Linear(hidden1_dim, feature_dim),  # layer 7
            # nn.Dropout(0.25),
            nn.LeakyReLU()
        )
        # stage 2:from features get classes
        # self.linear4 = nn.Linear(64, 512)
        # self.linear5 = nn.Linear(512, 512)
        # self.linear6 = nn.Linear(512, 256)
        # self.out_layer = nn.Linear(256, 16)
        # # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.25)
        self.category_layer = nn.Sequential(
            nn.Linear(feature_dim, hidden2_dim),  # layer 1
            # nn.Dropout(0.25),
            nn.LeakyReLU(),

            nn.Linear(hidden2_dim, hidden2_dim),  # layer 2
            # nn.Dropout(0.25),
            nn.LeakyReLU(),

            # nn.Linear(hidden2_dim, hidden2_dim),  # layer 3
            # nn.Dropout(0.25),
            # nn.LeakyReLU(),
            # nn.Linear(hidden2_dim, hidden2_dim),  # layer 4
            # # nn.Dropout(0.25),
            # nn.LeakyReLU(),
            # nn.Linear(hidden2_dim, hidden2_dim),  # layer 5
            # # nn.Dropout(0.25),
            # nn.LeakyReLU(),
            # nn.Linear(hidden2_dim, hidden2_dim),  # layer 6
            # # nn.Dropout(0.25),
            # nn.LeakyReLU(),
            nn.Linear(hidden2_dim, output_dim),  # layer 7
            # nn.Dropout(0.25),
            nn.Softmax()
        )

    def forward(self, x):
        # # stage 1
        # x = self.linear1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.linear2(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.linear0(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.linear0(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.linear3(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.feature_layer(x)
        # x = F.relu(x)
        #
        # # stage 2
        # x = self.linear4(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.linear5(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.linear6(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.out_layer(x)
        # x = F.relu(x)
        # output = F.log_softmax(x, dim=1)
        x = self.feature_layer(x)
        output = self.category_layer(x)

        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)  # 这里修改loss

        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} )'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
        if args.dry_run:
            break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = True # not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    # dataset1 = datasets.MNIST('./data', train=True, download=True,
    #                    transform=transform)
    # dataset2 = datasets.MNIST('./data', train=False,
    #                    transform=transform)
    # train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    # 自己做training set
    root = [os.getcwd() + '\\mydataset\\twitter_train.txt', os.getcwd() + '\\mydataset\\twitter_VP_train.txt',
            os.getcwd() + '\\mydataset\\twitter_SecondLady_train.txt']
    train_data = TwitterDataset(txts=root, transform=None)
    # test_data = MyDataset(txt=root + '_test.txt', transform=transforms.ToTensor())
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=4)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    test_data = TwitterDataset(txts=root, transform=None)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=4)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        # test按照全都正确来得到loss
        # test(model, device, test_loader)
        scheduler.step()
        test(model, device, test_loader)

    # test


    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
