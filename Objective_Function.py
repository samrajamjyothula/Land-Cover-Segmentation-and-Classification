import numpy as np
from Global_vars import Global_vars
from Segmentation_Evaluation import Segmentation_Evaluation
from TransResUnet_Fcn import TransResUnet_Fcn

def Objfun_Cls(Soln):
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Img = Global_vars.Feat
        Gt = Global_vars.GT
        per = round(Img.shape[0] * 0.75)
        train_data = Img[:per]
        Test_Data = Img[per:]
        train_target = Gt[:per]
        pred = TransResUnet_Fcn(train_data, train_target, Test_Data, sol.astype('int8'))
        Eval = Segmentation_Evaluation(train_data, pred)
        Fitn[i] = (1 /(Eval[1]+Eval[2]))
    return Fitn
