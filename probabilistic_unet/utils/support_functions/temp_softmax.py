import torch.nn as nn

class TempSoftmax(nn.Module):
    def __init__(self, temperature, dim=1):
        super(TempSoftmax, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, inp):
        scaled_logits = inp / self.temperature
        softmax_output = self.softmax(scaled_logits)
        return softmax_output