from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from util import util
import mnist_cnn, embed_net
from mnist_cnn import MNIST_CNN_Net
from embed_net import EmbedNet
from torchsummary import summary
import setting

# Training settings
def main():
    skipCNNTrainStage = True


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
    print("Cuda enabled? "+str(device))
    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    # Setting MNIST_CNN
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = MNIST_CNN_Net().to(device)
    summary(model,(1,28,28))
    if not skipCNNTrainStage:
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        #step1: let entropy shrink
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            curr_entropy = mnist_cnn.train(args, model, device, train_loader, optimizer, epoch)
            print("Curr entropy: "+str(curr_entropy))
            mnist_cnn.test(model, device, test_loader)
            scheduler.step()
    else:
        pretrained_dict = torch.load('mnist_cnn.pt')
        model1_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model1_dict}
        model1_dict.update(pretrained_dict)
        model.load_state_dict(model1_dict)
        model = model.cuda()

    #step2: data hiding using above model

    dict = model.state_dict()
    # params = model.named_parameters() #returns a generator
    carrier_weights = dict['conv2.weight'].cpu().numpy() # Size: 64*32*3*3
    modified_weights = [0]*(64*32*3*3)
    index = 0
    for i in range(64):
        for j in range(32):
            for k in range(3):
                for l in range(3):
                    modified_weights[index] = carrier_weights[i][j][k][l]
                    index+=1
    weight_num = 64*32*3*3
    watermark = [1.0]*setting.water_len
    modified_weights.append(watermark)
    #step2: EmbedNet - hiding data via NN
    model2 = EmbedNet.to(device,weight_num+setting.water_len,weight_num+setting.water_len)
    if not skipCNNTrainStage:
        optimizer = optim.Adadelta(model2.parameters(), lr=args.lr)

        #step1: let entropy shrink
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            curr_entropy = mnist_cnn.train(args, model, device, train_loader, optimizer, epoch)
            print("Curr entropy: "+str(curr_entropy))
            mnist_cnn.test(model, device, test_loader)
            scheduler.step()
    else:
        pretrained_dict = torch.load('mnist_cnn.pt')
        model1_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model1_dict}
        model1_dict.update(pretrained_dict)
        model.load_state_dict(model1_dict)
        model = model.cuda()

    #Save Models
    # if args.save_model:

    torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()