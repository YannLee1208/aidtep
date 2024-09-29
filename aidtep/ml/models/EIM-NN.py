import torch
from torch import nn

from aidtep.ml.models import ModelRegistry


class EIM_NN(nn.Module, ModelRegistry):

    @classmethod
    def name(cls):
        return "EIM-NN"

    def __init__(self, Q: torch.Tensor):
        """
        :param Q: shape (basis_num, full_dim)
        """
        super(EIM_NN, self).__init__()
        input_dim = 162

        self.Q = Q
        self.basis_number = Q.shape[0]
        activation_type = nn.Tanh

        self.fc = nn.Linear(input_dim, 512, dtype=Q.dtype)
        self.ac = activation_type()
        self.fc2 = nn.Linear(512, self.basis_number, dtype=Q.dtype)

    def forward(self, x):
        output = self.ac(self.fc(x))
        output = self.fc2(output)
        alpha = output.clone()

        output = output @ self.Q
        return output, alpha


if __name__ == "__main__":
    import numpy as np

    path = "../../../data/extract_basis/NOAA/SVD_base_number_10.npz"
    Q = np.load(path)
    Q = torch.tensor(Q["basis"])

    model = EIM_NN(Q)

    x = torch.randn(10, 162).double()
    output, alpha = model(x)
    print(output.shape)
    print(alpha.shape)
