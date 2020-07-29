from __future__ import print_function
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from util import util
from torch.utils.data import Dataset, DataLoader
from mydataset.RandSequence import RandSeqDataset
import setting
from MyEntropyLoss import my_entropy_loss
from MyEntropyLoss import my_entropy_loss
lamb = 0.01 # 0.001 maybe is too small,loss is usually 0.006

# with open('config') as json_file:
#     opt = json.load(json_file)
# opt['manual_seed']

# target_entropy = 6.60
class RandEntroNet(nn.Module):
    def __init__(self,input_dim,hidden1_dim):
        super(RandEntroNet, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),  # layer 1
            nn.LeakyReLU(),
            nn.Linear(hidden1_dim, hidden1_dim),  # layer 2
            nn.LeakyReLU(),
            nn.Linear(hidden1_dim, input_dim),  # layer 2
            nn.LeakyReLU(),
        )

    def forward(self, x):
        output = self.feature_layer(x)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    entropy = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)  # 这里修改loss

        ### Loss: Let Entropy Shrink  2020.7.29
        # dict = model.state_dict()
        # carrier_weights = dict['conv2.weight'].cpu().numpy()
        # histogram = util.get_histogram(carrier_weights)
        # entropy = util.entropy_calculate(histogram)
        # loss_entropy = lamb * entropy
        #Entropy range:(6.6~7.0)


        entropy = [0]*len(target)
        for i in range(len(output)):
            histogram = util.get_histogram(output.clone().cpu().detach().numpy()[i])
            entropy[i] = util.entropy_calculate(histogram)
        entroLoss = F.mse_loss(torch.tensor(entropy).cuda(),target).item()

        loss = (1+F.mse_loss(output,output))*entroLoss
        ###
        # loss = my_entropy_loss().forward(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Entropy Ave: {:2.4f})'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),sum(entropy)/len(entropy)))
            if args.dry_run:
                break

    return entropy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            entropy = [0] * len(target)
            for i in range(len(output)):
                histogram = util.get_histogram(output.clone().cpu().detach().numpy()[i])
                entropy[i] = util.entropy_calculate(histogram)
            loss = F.mse_loss(torch.tensor(entropy).cuda(), target)
            test_loss += loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += sum([1 if abs(entropy[i]-target[i])<0.1 else 0 for i in range(len(target))])

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def self_defined_entropy_loss(output,target_entropy):
    # 期望output本身满足熵为entropy
    histogram = util.get_histogram(output)
    entropy = util.entropy_calculate(histogram)
    return abs(entropy-target_entropy)

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
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
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Cuda enabled? " + str(device))
    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )

    train_data = RandSeqDataset(sequence_len=setting.carrier_weight_num)
    test_data = RandSeqDataset(sequence_len=setting.carrier_weight_num)
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True, num_workers=4)

    model = RandEntroNet(setting.carrier_weight_num,setting.carrier_weight_num).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # step1: let entropy shrink
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        curr_entropy = train(args, model, device, train_loader, optimizer, epoch)
        print("Curr entropy: " + str(curr_entropy))
        test(model, device, test_loader)
        scheduler.step()

if __name__ == '__main__':
    main()
