import importlib

from torch import nn, optim
from torch.optim.lr_scheduler import StepLR


def get_model_class(model_type):
    module_path = f"aidtep.ml.models.{model_type}"
    module = importlib.import_module(module_path)
    model_class = getattr(module, model_type)
    return model_class


# TODO: Implement the get_criterien function
def get_criterien(criteria_type: str):
    return nn.MSELoss
    module_path = f"aidtep.ml.criteria.{criteria_type}"
    module = importlib.import_module(module_path)
    criteria_class = getattr(module, criteria_type)
    return criteria_class


# TODO: Implement the get_optimizer function
def get_optimizer(optimzer_type: str):
    return optim.Adam


# TODO: Implement the get_scheduler function
def get_scheduler(scheduler_type: str):
    return StepLR

if __name__ == '__main__':
    model = get_model_class("NVT_ResNet")
    print(model)
