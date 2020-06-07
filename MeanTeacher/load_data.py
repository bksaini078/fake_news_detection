import os
import numpy as np

def loading_data(args):
    if not os.path.isfile(args.inputPath + args.xTest):
        print("Please clean the data first")
    else:
        x_tr = np.load(args.inputPath + args.xTrain,allow_pickle=True)
        x_te = np.load(args.inputPath + args.xTest ,allow_pickle=True)
        y_tr = np.load(args.inputPath + args.yTrain,allow_pickle=True)
        y_te = np.load(args.inputPath + args.yTest,allow_pickle=True)
        x_un = np.load(args.inputPath + args.xUnlabel,allow_pickle=True)

    return x_tr, y_tr, x_te, y_te, x_un

def add_params_data(parser):
    parser.add_argument('--inputPath',type=str, default=os.path.abspath(os.getcwd())+"\\Data\\Input\\")
    parser.add_argument('--outputPath',type=str, default=os.path.abspath(os.getcwd())+"\\Data\\Output\\")
    parser.add_argument('--xTrain', type=str,default="xtr_shuffled.npy")
    parser.add_argument('--xTest', type=str,default="xte_shuffled.npy")
    parser.add_argument('--yTrain', type=str,default="ytr_shuffled.npy")
    parser.add_argument('--yTest', type=str,default="yte_shuffled.npy")
    parser.add_argument('--xUnlabel', type=str,default="xun_shuffled.npy")

    return parser
