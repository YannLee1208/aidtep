from torch import optim

from aidtep.ml.optimizer import OptimizerRegistry


class AdamWrapper(optim.Adam, OptimizerRegistry):
    @classmethod
    def name(cls):
        return 'adam'

    def __init__(self, *args, **kwargs):
        super(AdamWrapper, self).__init__(*args, **kwargs)