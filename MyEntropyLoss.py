import torch
from util import util
import torch.nn.functional as F


class my_entropy_loss(torch.nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(my_entropy_loss, self).__init__()

    def forward(self, output, target):
        """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        """
        # hinge_loss = 1 - torch.mul(output, target)
        # hinge_loss[hinge_loss < 0] = 0
        # 不要忘记返回scalar
        # return torch.mean(hinge_loss)

        entropy = [0] * len(target)
        for i in range(len(output)):
            histogram = util.get_histogram(output.clone().cpu().detach().numpy()[i])
            entropy = util.entropy_calculate(histogram)
        loss = F.mse_loss(torch.tensor(entropy).cuda(), target)

        return loss

