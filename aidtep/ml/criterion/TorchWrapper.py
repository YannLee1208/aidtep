from torch.nn import MSELoss

from aidtep.ml.criterion import CriterionRegistry


class MSELossWrapper(MSELoss, CriterionRegistry):

    @classmethod
    def name(cls):
        return "mse"

    def __init__(self, *args, **kwargs):
        super(MSELoss, self).__init__(*args, **kwargs)

