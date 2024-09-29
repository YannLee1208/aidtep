import numpy as np
import torch
from torch import nn

from aidtep.ml.models import ModelRegistry


class EIM_NN(nn.Module, ModelRegistry):

    @classmethod
    def name(cls):
        return "EIM-NN"

    def __init__(self, basis_path: str, x_shape: int, y_shape: int):
        """
        :param basis_path: path to the basis file, npz file, with key "basis". Q is the basis matrix, shape is (basis_number, full_dim)
        """
        super(EIM_NN, self).__init__()
        input_dim = 162
        self.x_shape = x_shape
        self.y_shape = y_shape

        basis_res = np.load(basis_path)
        self.Q = torch.tensor(basis_res["basis"])
        self.basis_number = self.Q.shape[0]
        activation_type = nn.Tanh

        self.fc = nn.Linear(input_dim, 512, dtype=self.Q.dtype)
        self.ac = activation_type()
        self.fc2 = nn.Linear(512, self.basis_number, dtype=self.Q.dtype)

    def forward(self, x):
        self.Q = self.Q.to(x.device)
        output = self.ac(self.fc(x))
        output = self.fc2(output)
        # alpha = output.clone()

        output = output @ self.Q
        output = output.view(-1, self.x_shape, self.y_shape)
        return output


if __name__ == "__main__":

    path = "../../../data/extract_basis/NOAA/SVD_base_number_10.npz"

    model = EIM_NN(path)

    x = torch.randn(10, 162).double()
    output, alpha = model(x)
    print(output.shape)
    print(alpha.shape)
