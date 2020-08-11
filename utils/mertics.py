import torch.nn as nn
import torch


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.eps = eps

    def forward(self, y, y_pred):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(y, y_pred) + self.eps)
        return loss


class MAPELoss(nn.Module):
    def __init__(self, eps=1):
        super(MAPELoss, self).__init__()
        self.eps = eps

    def forward(self, y, y_pred):
        # y_indices = y.nozero()
        # loss = torch.zeros()
        # for i in y_indices:
        #     relative_error = torch.mean(torch.abs(y[i] - y_pred[i]) / y[i])
        #     loss += relative_error
        # loss = loss / len(y_indices)
        loss = torch.mean(torch.abs(y - y_pred) / (y + self.eps))
        return loss
