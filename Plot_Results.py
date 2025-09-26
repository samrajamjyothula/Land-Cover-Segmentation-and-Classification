import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import seaborn as sns
import matplotlib
from itertools import cycle
from sklearn.metrics import roc_curve, confusion_matrix

def Statistical(val):
    out = np.zeros((5))
    out[0] = max(val)
    out[1] = min(val)
    out[2] = np.mean(val)
    out[3] = np.median(val)
    out[4] = np.std(val)
    return out

def Plot_Image_Results():
    for a in range(4):
        eval = np.load('Eval_seg.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'Dice', 'Jaccard']
        value = eval[ :, :, :]
        stat = np.zeros((value.shape[1], value.shape[2], 5))
        for j in range(value.shape[1]): # For all algms and Mtds
            for k in range(value.shape[2]): # For all terms
                stat[j, k, :] = Statistical(value[:, j, k])
        stat = stat
        for k in range(len(Terms)):
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, stat[0, k, :], color='#f97306', width=0.10, label="TSO-AHCNN")
            ax.bar(X + 0.10, stat[1, k, :], color='#f10c45', width=0.10, label="BWO-AHCNN")
            ax.bar(X + 0.20, stat[2, k, :], color='#ddd618', width=0.10, label="CO-AHCNN")
            ax.bar(X + 0.30, stat[3, k, :], color='#6ba353', width=0.10, label="HBA-AHCNN")
            ax.bar(X + 0.40, stat[4, k, :], color='#13bbaf', width=0.10, label="IFHBA-AHCNN")
            plt.xticks(X + 0.10, ('Best', 'Worst', 'Mean', 'Median', 'Std'))
            plt.ylabel(Terms[k])
            plt.xlabel('Statistical Analysis')
            plt.legend(loc=1)
            plt.tight_layout()
            path1 = "./Results/Segmented_Image_%s_alg_%s.png" % (str(a + 1), Terms[k])
            plt.savefig(path1)
            plt.show()


            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, stat[5, k, :], color='#f97306', width=0.10, label="Unet")
            ax.bar(X + 0.10, stat[6, k, :], color='#cc3f81', width=0.10, label="Res-Unet")
            ax.bar(X + 0.20, stat[7, k, :], color='#ccbc3f', width=0.10, label="Trans-ResUnet")
            ax.bar(X + 0.30, stat[8, k, :], color='c', width=0.10, label="AHCNN")
            ax.bar(X + 0.40, stat[9, k, :], color='k', width=0.10, label="IFHBA-AHCNN")
            plt.xticks(X + 0.10, ('Best', 'Worst', 'Mean', 'Median', 'Std'))
            plt.ylabel(Terms[k])
            plt.xlabel('Statistical Analysis')
            plt.legend(loc=1)
            path1 = "./Results/Segmented_Image_%s_bar_%s.png" % (str(a+1),Terms[k])
            plt.tight_layout()
            plt.savefig(path1)
            plt.show()

def plot_results_activation():
    for a in range(4):
        Eval = np.load('Eval_all.npy', allow_pickle=True)[a]
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
        Graph_Term = [0,3,4,5,9]
        Algorithm = ['TERMS', 'EOO', 'DO', 'LO', 'HBA', 'PROPOSED']
        Classifier = ['TERMS', 'LSTM', 'RNN', 'GRU', 'Resnet', 'AHCNN-RAN']
        value = Eval[4, :, 4:]
        value[:, :-1] = value[:, :-1] * 100
        value[9,:] = value[ 4, :]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('--------------------------------------------------Dataset-'+str(a+1)+'-Classifier Comparison - ',
              'Activation Function --------------------------------------------------')
        print(Table)
        Eval = np.load('Eval_all.npy', allow_pickle=True)[a]
        learnper = [1, 2, 3, 4, 5]
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    if Graph_Term[j] == 9:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4] * 100
            Graph[:, 9] = Graph[:, 4]

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.1, 0.7, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='#f97306', edgecolor='k', width=0.10, hatch="+", label="LSTM")
            ax.bar(X + 0.10, Graph[:, 6], color='#f10c45', edgecolor='k', width=0.10, hatch="x", label="RNN")
            ax.bar(X + 0.20, Graph[:, 7], color='#ddd618', edgecolor='k', width=0.10, hatch="/", label="GRU")
            ax.bar(X + 0.30, Graph[:, 8], color='#6ba353', edgecolor='k', width=0.10, hatch="o", label="Resnet")
            ax.bar(X + 0.40, Graph[:, 9], color='#13bbaf', edgecolor='r', width=0.10, hatch="*",label="AHCNN-RAN")
            plt.xticks(X + 0.10, ('1', '2','3',  '4','5'), rotation=7)
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_kfold_%s_bar_lrean.png" % (a + 1, Terms[Graph_Term[j]])
            plt.xlabel('KFold')
            plt.tight_layout()
            plt.savefig(path1)
            plt.show()

def Confusion_matrix():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    Eval = np.load('Eval_all.npy',allow_pickle= True)
    no_of_Dataset = 4
    for n in range(no_of_Dataset):
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[n][4]).argmax(axis=1), np.asarray(Predict[n][4]).argmax(axis=1))
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax).set(title='Accuracy = ' + str(Eval[n,4, 4, 4] * 100)[:5] + '%')
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.savefig(path)
        plt.show()

def Plot_ROC():
    lw = 2
    cls = ['LSTM', 'RNN', 'GRU', 'RESNET', 'AHCNN-RAN']
    colors1 = cycle(["plum", "red", "palegreen", "chocolate", "navy", ])
    colors2 = cycle(["hotpink", "plum", "chocolate", "navy", "red", "palegreen", "violet", "red"])
    for n in range(4):
        for i, color in zip(range(5), colors1):  # For all classifiers
            Predicted = np.load('roc_score.npy', allow_pickle=True)[n][i].astype('float')
            Actual = np.load('roc_act.npy', allow_pickle=True)[n][i].astype('int')
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label="{0}".format(cls[i]),
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/_roc_%s.png"  %(str(n+1))
        plt.tight_layout()
        plt.savefig(path1)
        plt.show()

def Fitness():
    for a in range(4):
        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['TSO-AHCNN', 'BWO-AHCNN', 'CO-AHCNN', 'HBA-AHCNN', 'IFHBA-AHCNN']

        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('--------------------------------------------------Dataset_'+str(a+1)+'Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.plot(iteration, conv[0, :], color='r', linewidth=3, marker='>', markerfacecolor='blue', markersize=8,
                 label="TSO-AHCNN")
        plt.plot(iteration, conv[1, :], color='g', linewidth=3, marker='>', markerfacecolor='red', markersize=8,
                 label="BWO-AHCNN")
        plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='>', markerfacecolor='green', markersize=8,
                 label="CO-AHCNN")
        plt.plot(iteration, conv[3, :], color='m', linewidth=3, marker='>', markerfacecolor='yellow', markersize=8,
                 label="HBA-AHCNN")
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='>', markerfacecolor='cyan', markersize=8,
                 label="IFHBA-AHCNN")
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path1 = "./Results/conv_%s.png"  %(str(a+1))
        plt.tight_layout()
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    Plot_Image_Results()
    plot_results_activation()
    Confusion_matrix()
    Plot_ROC()
    Fitness()


