from load_data import loading_data
from tokenization import tokenization
from Model.biLSTM import BiLstmModel
import tensorflow as tf
from ConfusionMatrix import Confusion_matrix
from Reports.ReportWriting import report_writing


def add_parms_supervised(parser):
    parser.add_argument('--maxlen', default=300, type=int)
    return parser


def train_supervised(args):
    # maxlen = 300
    # batch_size = 64
    # loading data
    x_train, y_train, x_test, y_test, x_unlabel = loading_data(args)
    
    # data tokenisation is pending here
    x_train, x_test, x_unlabel, vocab_size = tokenization(args, x_train,x_test, x_unlabel, args.maxlen)

    model_supervised = BiLstmModel(args.maxlen, vocab_size)
    model_supervised.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= args.lr ),loss= 'binary_crossentropy', metrics=['accuracy'])
    print('Training supervised Model...')
    history=model_supervised.fit(x_train, y_train,batch_size=args.batchSize,epochs=args.epochs,validation_split=0.2)
    # evaluation
    train_accuracy=history.history['accuracy'][len(history.epoch)-1]
    cm, test_accuracy, precision, recall, f1_score =Confusion_matrix(model_supervised,x_test,y_test,args.threashold, 'Supervised model')
    report_writing(args, 'Supervised_BILSTM', args.batchSize,len(history.epoch),'NaN','NaN', train_accuracy,test_accuracy, f1_score, precision,recall,'Labelled data')   

