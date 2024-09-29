from aidtep.inverse_problem import InverseBuilder
from aidtep.ml.criterion import get_criterion_class
from aidtep.ml.models import get_model_class
from aidtep.ml.optimizer import get_optimizer_class


class IAEAInverseBuilder(InverseBuilder):
    @classmethod
    def name(cls):
        return "IAEA"


if __name__ == "__main__":
    # initialize(log_level=logging.DEBUG)
    # import torch
    # print(torch.cuda.is_available())

    # get_model_class("NVT_ResNet")
    loss = get_criterion_class("l2")()
    model = get_model_class("NVT_ResNet")()
    optimizer = get_optimizer_class("adam")(model.parameters(), lr=0.001)
    import torch

    print(loss(torch.Tensor([1, 2, 3]), torch.Tensor([1, 2, 2])))
    print(optimizer)
