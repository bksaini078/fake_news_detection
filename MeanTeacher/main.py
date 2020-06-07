import argparse

from train_supervised import add_parms_supervised, train_supervised
from load_data import loading_data, add_params_data
from Reports.ReportWriting import add_params_report
from meanTeacher import train_MeanTeacher, add_params_mt

if __name__ == '__main__':
    # parameters from arugument parser 
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0001, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batchSize', default=64, type=int)
    parser.add_argument('--ratio', default=0.5, type=int)
    parser.add_argument('--alpha', type=int, default=0.97)
    parser.add_argument('--threashold', type=int, default=0.51)

    # Adding parameters from other python files
    parser = add_parms_supervised(parser)
    parser = add_params_data(parser)
    parser = add_params_report(parser)
    parser = add_params_mt(parser)
    args = parser.parse_args()

    for i in range(0,1):
        args.alpha = args.alpha+0.01
        for i in range(0,1):
            train_supervised(args)
            train_MeanTeacher(args)
    print('finished')