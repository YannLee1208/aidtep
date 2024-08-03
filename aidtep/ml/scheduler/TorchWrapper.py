from torch import optim

from aidtep.ml.scheduler import SchedulerRegistry


class StepLRWrapper(optim.lr_scheduler.StepLR, SchedulerRegistry):
    @classmethod
    def name(cls):
        return 'step_lr'

    def __init__(self, *args, **kwargs):
        super(StepLRWrapper, self).__init__(*args, **kwargs)