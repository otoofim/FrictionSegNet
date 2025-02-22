import torch
import torch.nn as nn


class MyMSE(torch.nn.Module):
    def __init__(self):
        super(MyMSE, self).__init__()

    def forward(self, output, target):
        mask = target.detach() != 0
        loss = torch.pow(target - output, 2)
        loss = loss * mask
        loss_mean = loss.sum() / (mask.sum() + 1e-15)

        return loss_mean


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


class CrossEntopy(torch.nn.Module):
    def __init__(self, label_smoothing):
        super(CrossEntopy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(
            reduction="none", label_smoothing=label_smoothing
        )

    def forward(self, output, target):
        CEL = torch.mean(self.criterion(input=output, target=target), (1, 2))
        return CEL
