
from torch.nn import Module, Linear
import torch.nn.functional as F

class MnistModel(Module):

    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = Linear(28 * 28, 50)
        self.fc2 = Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        y = F.log_softmax(self.fc2(x), dim=1)
        return y