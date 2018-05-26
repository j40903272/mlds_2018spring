from torch import nn

class Stack(nn.Module):
    def __init__(self, num, func):
        super().__init__()
        layers = []
        for i in range(num):
            l = func(i)
            if isinstance(l, nn.Module):
                layers.append(func(i))
            else:
                layers.append(nn.Sequential(*func(i)))

        self.layer = nn.Sequential(*layers)
        self.forward = self.layer.forward
