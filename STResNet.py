import torch
import torch.nn as nn
import time
import os
torch.manual_seed(1)


class ResUnit(nn.Module):
    """
    left: (BN + ReLU + Convolution) * 2
    residual: shortcut
    """

    def __init__(self, filter_num):
        super(ResUnit, self).__init__()
        self.left = nn.Sequential(
            nn.BatchNorm2d(filter_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(filter_num),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filter_num, out_channels=filter_num, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        out = self.left(x)
        residual = x  # shortcut
        out += residual
        return out


class Conv1ResUnitsConv2(nn.Module):
    """
    Conv1 + ResUnit * 4 + Conv2
    """

    def __init__(self, in_dim, filter_num=64, L=4):
        super(Conv1ResUnitsConv2, self).__init__()
        self.L = L
        self.Conv1 = nn.Conv2d(in_dim, filter_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.ResUnits = self._stack_resunits(filter_num)
        # in-flow & out-flow --> out_channel=2
        self.Conv2 = nn.Conv2d(filter_num, 2, kernel_size=3, stride=1, padding=1, bias=True)

    def _stack_resunits(self, filter_num):
        layers = []
        for i in range(0, self.L):
            layers.append(ResUnit(filter_num))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.ResUnits(x)
        x = self.Conv2(x)
        return x


class STResNet(nn.Module):
    def __init__(self, lc=3, lp=3, lt=3, flow_dim=2, map_height=32, map_width=32, external_dim=28):
        super(STResNet, self).__init__()
        self.lc = lc
        self.lp = lp
        self.lt = lt
        self.flow_dim = flow_dim
        self.map_height = map_height
        self.map_width = map_width
        self.external_dim = external_dim

        self.branch_c = Conv1ResUnitsConv2(in_dim=self.lc * self.flow_dim)
        self.branch_p = Conv1ResUnitsConv2(in_dim=self.lp * self.flow_dim)
        self.branch_t = Conv1ResUnitsConv2(in_dim=self.lt * self.flow_dim)

        if self.external_dim is not None and self.external_dim > 0:
            self.external_ops = nn.Sequential(
               nn.Linear(in_features=self.external_dim, out_features=10, bias=True),
               nn.ReLU(inplace=True),
               nn.Linear(in_features=10, out_features=self.flow_dim * self.map_height * self.map_width, bias=True),
               nn.ReLU(inplace=True)
            )

        self.weight_c = nn.Parameter(torch.randn(1, self.flow_dim, self.map_height, self.map_width))
        self.weight_p = nn.Parameter(torch.randn(1, self.flow_dim, self.map_height, self.map_width))
        self.weight_t = nn.Parameter(torch.randn(1, self.flow_dim, self.map_height, self.map_width))

        self.model_name = str(type(self).__name__)

    def forward(self, x_c, x_p, x_t, x_ext):
        x_c = x_c.view(-1, self.lc * self.flow_dim, self.map_height, self.map_width)
        x_p = x_p.view(-1, self.lp * self.flow_dim, self.map_height, self.map_width)
        x_t = x_t.view(-1, self.lt * self.flow_dim, self.map_height, self.map_width)
        c_out = self.branch_c(x_c)
        p_out = self.branch_p(x_p)
        t_out = self.branch_t(x_t)
        # parameter-matrix-based fusion
        main_out = self.weight_c * c_out + self.weight_p * p_out + self.weight_t * t_out

        if self.external_dim is not None and self.external_dim > 0:
            ext_out = self.external_ops(x_ext)
            ext_out = ext_out.view(-1, self.flow_dim, self.map_height, self.map_width)
            main_out += ext_out

        main_out = torch.tanh(main_out)
        return main_out

    def save(self, stop_epoch, name=None):
        prefix = 'checkpoints' + os.sep
        if name is None:
            prefix = prefix + self.model_name + "_"
            name = time.strftime(prefix + '%m%d_%H_%M_' + str(stop_epoch) + '.pth')
        else:
            name = prefix + name + '.pth'
        torch.save(self.state_dict(), name)
        print("Training finished, the STResNet instance ["+str(name)+"] has been saved.")

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print("The training model was successfully loaded.")


















