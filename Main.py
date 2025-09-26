import os
import random as rn
from scipy.io import loadmat
from BWO import BWO
from Batch_Split import Batch_Split
from CO import CO
from Global_vars import Global_vars
from HBA import HBA
from Image_Result import Image_Results
from Model_GRU import Model_GRU
from Model_LSTM import Model_LSTM
from Model_RAN import Model_RAN
from Model_RESNET import Model_RESNET
from Model_RNN import Model_RNN
from Objective_Function import Objfun_Cls
from PROPOSED import PROPOSED
from Plot_Results import *
from Resunet import Resunet
from TSO import TSO
from TransResUnet import TransResUnet
from TransResUnet_Fcn import TransResUnet_Fcn
from Unet import Unet

no_of_datasets = 4

def Read_Dataset(Directory):
    listDataset = os.listdir(Directory)
    for i in range(len(listDataset)):
        indir = Directory + listDataset[i]
        listFiles = os.listdir(indir)
        for j in range(len(listFiles)):
            filename = indir + '/' + listFiles[j]
            if "gt" in listFiles[j]:
                data = loadmat(filename)
                structName = listFiles[j].split('.')[0]
                structName = structName[0].lower() + structName[1:]
                GT = data[structName]
                np.save('GroundTruth_' + str(i + 1) + '.npy', GT)
            else:
                data = loadmat(filename)
                structName = listFiles[j].split('.')[0]
                structName = structName[0].lower() + structName[1:]
                Image = data[structName]
                Images = []
                for k in range(Image.shape[2]):
                    im = Image[:, :, k]
                    Images.append(im)
                np.save('Images_' + str(i + 1) + '.npy', Images)

# Read  All Datasets
an = 0
if an == 1:
    Read_Dataset('./Datasets/')

# Patch splitting
an = 0
if an == 1:
    for i in range(no_of_datasets):
        batch_img = []
        Images = np.load('Images_' + str(i + 1) + '.npy', allow_pickle=True)
        for j in range(Images.shape[0]):
            image = Images[j]
            patches = Batch_Split(image)
            batch_img.append(patches)
        np.save('SplittedImages_' + str(i + 1) + '.npy', batch_img)

# Ground Truth
an = 0
if an == 1:
    for i in range(no_of_datasets):
        Images = np.load('GroundTruth_' + str(i + 1) + '.npy', allow_pickle=True)
        patche = Batch_Split(Images)
        Target = []
        for j in range(len(patche)):
            gr_tru = patche[j]
            uniq = np.unique(gr_tru)
            lenUniq = [len(np.where(gr_tru == uniq[k])[0]) for k in range(len(uniq))]
            maxIndex = np.where(lenUniq == np.max(lenUniq))[0][0]
            target = uniq[maxIndex]
            Target.append(target)
        Targ = np.asarray(Target)
        uni = np.unique(Targ)
        tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
        for a in range(len(uni)):
            ind = np.where((Targ == uni[a]))
            tar[ind[0], i] = a
        np.save('Target_' + str(i + 1) + '.npy', tar)
        np.save('SplittedGroundTruth_' + str(i + 1) + '.npy', patche)

# Optimization for Segmentation
an = 0
if an == 1:
    Best_sol = []
    fitness = []
    for n in range(no_of_datasets):
        Images = np.load('SplittedImages_' + str(n + 1) + '.npy', allow_pickle=True)
        GroundTruth = np.load('SplittedGroundTruth_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_vars.Feat = Images
        Global_vars.GT = GroundTruth
        Npop = 10
        Chlen = 4
        xmin = np.asarray([5, 50, 5, 50]) * np.ones((Npop, Chlen))
        xmax = np.asarray([255, 100, 255, 100]) * np.ones((Npop, Chlen))
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])
        fname = Objfun_Cls
        max_iter = 50

        print('TSO....')
        [bestfit1, fitness1, bestsol1, Time1] = TSO(initsol, fname, xmin, xmax, max_iter)

        print('BWO')
        [bestfit2, fitness2, bestsol2, Time2] = BWO(initsol, fname, xmin, xmax, max_iter)

        print('CO....')
        [bestfit3, fitness3, bestsol3, Time3] = CO(initsol, fname, xmin, xmax, max_iter)

        print('HBA....')
        [bestfit4, fitness4, bestsol4, Time4] = HBA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

        sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        fit = [fitness1.ravel(),fitness2.ravel(),fitness3.ravel(),fitness4.ravel(),fitness5.ravel()]
        Best_sol.append(sol)
        fitness.append(fit)
    np.save('Best_sol.npy', Best_sol)
    np.save('Fitness.npy',fitness)

# Optimized for segmentation
an = 0
if an == 1:
    for i in range(4):
        Images = np.load('SplittedImages_' + str(i + 1) + '.npy', allow_pickle=True)
        GT = np.load('SplittedGroundTruth_' + str(i + 1) + '.npy', allow_pickle=True)
        sol = np.load('Best_sol.npy', allow_pickle=True)[i][4, :]
        per = round(Images.shape[0] * 0.75)
        train_data = Images[:per]
        Test_Data = Images
        train_target = GT[:per]
        Image = TransResUnet_Fcn(train_data, train_target, Test_Data,sol)
        np.save('IFHBA-AHCNN_'+ str(i + 1) + '.npy',Image)

# Classification
an = 0
if an == 1:
    Eval_all = []
    for i in range(no_of_datasets):
        Feat = np.load('IFHBA-AHCNN_' + str(i + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(i + 1) + '.npy', allow_pickle=True)
        Activation_function = [1,2,3,4,5]
        EVAL1 = []
        for m in range(5):
            per = round(len(Feat) * (Activation_function[m]))
            EVAL = np.zeros((5, 14))
            Train_Data = Feat[:per, :]
            Train_Target = Target[:per]
            Test_data = Feat[per:, :]
            Test_Target = Target[per:]
            EVAL[0, :] = Model_LSTM(Train_Data, Train_Target, Test_data, Test_Target)
            EVAL[1, :] = Model_RNN(Train_Data, Train_Target, Test_data, Test_Target)
            EVAL[2, :] = Model_GRU(Train_Data, Train_Target, Test_data, Test_Target)
            EVAL[3, :] = Model_RESNET(Train_Data, Train_Target, Test_data, Test_Target)
            EVAL[4, :] = Model_RAN(Train_Data, Train_Target, Test_data, Test_Target)
            EVAL1.append(EVAL)
        Eval_all.append(EVAL1)
    np.save('Eval_all.npy', Eval_all)

# Image_Comparison
an = 0
if an == 1:
    for i in range(4):
        Images = np.load('SplittedImages_' + str(i + 1) + '.npy', allow_pickle=True)
        GT = np.load('SplittedGroundTruth_' + str(i + 1) + '.npy', allow_pickle=True)
        per = round(Images.shape[0] * 0.75)
        train_data = Images[:per]
        Test_Data = Images
        train_target = GT[:per]
        Image1 = Unet(train_data, train_target, Test_Data)
        Image2 = Resunet(train_data, train_target, Test_Data)
        Image3 = TransResUnet(train_data, train_target, Test_Data)
        Image4 = TransResUnet_Fcn(train_data, train_target, Test_Data)
        np.save('Unet_' + str(i + 1) + ',npy', Image1)
        np.save('ResUnet_' + str(i + 1) + ',npy', Image1)
        np.save('TransResUnet_' + str(i + 1) + ',npy', Image1)
        np.save('AHCNN_' + str(i + 1) + ',npy', Image1)

Plot_Image_Results()
plot_results_activation()
Confusion_matrix()
Plot_ROC()
Fitness()
Image_Results()
