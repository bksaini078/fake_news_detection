import datetime
import pandas as pd
from pathlib import Path


def report_writing( args, Model,Batch_Size, Epoch,Alpha,Ratio, train_accuracy,test_accuracy, F1_Score,Precision,Recall, comment):
    x = datetime.datetime.now()
    report_df = pd.DataFrame(columns=['Date', 'Model','Batch_Size', 'Epoch','Alpha','Ratio','Train_Accuracy','Test_Accuracy', 'F1_Score','Precision','Recall', 'comment'])
    report_df = report_df.append({'Date' : x.strftime("%c"), 'Model' :Model,'Batch_Size' : Batch_Size, 'Epoch': Epoch,'Alpha':Alpha,'Ratio':Ratio,'Train_Accuracy':train_accuracy,'Test_Accuracy':test_accuracy, 'F1_Score':F1_Score,'Precision': Precision,'Recall':Recall,'comment':comment}, ignore_index=True)
    my_file = Path(args.outputPath + args.reportIterate)

    if my_file.exists():
        report_df.to_csv(args.outputPath + args.reportIterate,mode='a', header= False , index = False)
    else:
        report_df.to_csv(args.outputPath + args.reportIterate,mode='w', header= True , index= False) 
    return 

def add_params_report(parser):
    parser.add_argument('--reportIterate', type=str,default="report_iterate.csv")

    return parser