from modules.DCNO import DCNO2d
from modules.FNO import FNO2d
from modules.DilResNet import DilResNet
from modules.GT import GT
from modules.Unet import Unet
from modules.UNO import UNO, UNO_9
from modules.MWT import MWT2d
from modules.CNN import CNN

def get_model(model_name, R_dic):
    if model_name == 'DCNO':
        model = DCNO2d(R_dic)
    elif model_name == 'FNO':
        model = FNO2d(R_dic)
    elif model_name == 'DilResNet':
        model = DilResNet(R_dic)
    elif model_name == 'GT':
        model = GT(R_dic)
    elif model_name == 'Unet':
        model = Unet(R_dic)
    elif model_name == 'UNO_9':
        model = UNO_9(R_dic)
    elif model_name == 'UNO':
        model = UNO(R_dic)
    elif model_name == 'MWT':
        model = MWT2d(R_dic)
    elif model_name == 'CNN':
        model = CNN(R_dic)

    return model
