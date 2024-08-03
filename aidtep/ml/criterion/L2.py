import torch
from torch import nn

from aidtep.ml.criterion import CriterionRegistry


class L2Loss(nn.Module, CriterionRegistry):

    @classmethod
    def name(cls):
        return "l2"

    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, y_predict: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_predict.shape[0]
        errors = torch.zeros(batch_size, device=y_predict.device)
        for i in range(batch_size):
            y_true_tensor = y_true[i]
            y_pred_tensor = y_predict[i]
            errors[i] = torch.norm(y_pred_tensor - y_true_tensor) / torch.norm(y_true_tensor)
        return errors.mean()


class LinfLoss(nn.Module, CriterionRegistry):

    @classmethod
    def name(cls):
        return "linf"

    def __init__(self):
        super(LinfLoss, self).__init__()

    def forward(self, y_predict: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_predict.shape[0]
        errors = torch.zeros(batch_size, device=y_predict.device)
        for i in range(batch_size):
            y_true_tensor = y_true[i]
            y_pred_tensor = y_predict[i]
            errors[i] = torch.max(torch.abs(y_pred_tensor - y_true_tensor)) / torch.max(torch.abs(y_true_tensor))
        return errors.mean()