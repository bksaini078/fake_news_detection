from load_data import loading_data
from tokenization import tokenization
import tensorflow as tf
from Cost.cost import classification_costs, Consistency_Cost, Overall_Cost, ema
from Model.biLSTM import BiLstmModel
from NoiseCreater.instantNoise import instant_noise
from ConfusionMatrix import Confusion_matrix
from Plot.plot import scatter_plot, model_evaluation
from Reports.ReportWriting import report_writing

def train_MeanTeacher(args):
    #declaring hyper paramters
    
    # maxlen = 300
   
    # epochs = 6
    # alpha = 0.99
    
    

    # loading data
    x_train, y_train, x_test, y_test, x_unlabel = loading_data(args)
    # data tokenisation is pending here
    x_train, x_test, x_unlabel_tar, vocab_size = tokenization(args, x_train,x_test, x_unlabel, args.maxlen)
   
   # tensorslices

    #preparing the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(args.batchSize)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(args.batchSize)

    #preparing the target dataset 
    tar_dataset =  tf.data.Dataset.from_tensor_slices(x_unlabel_tar)
    tar_dataset = tar_dataset.shuffle(buffer_size=1024).batch(args.batchSize)

    #declaring optimiser
    optimizer= tf.keras.optimizers.Adam(learning_rate= args.lr ) #trying changing learning rate , sometimes it gives good result 
    train_metrics = tf.keras.metrics.BinaryAccuracy(name='Binary_Accuracy')
    val_acc_metric = tf.keras.metrics.BinaryAccuracy(name="Binary_Acc")
    teacher_acc_metric = tf.keras.metrics.BinaryAccuracy(name="Binary_Acc_teacher") 
    # Creating model
    student = BiLstmModel(args.maxlen, vocab_size)
    teacher = BiLstmModel(args.maxlen, vocab_size)


    # collecting costs
    #this one for collecting the costs
    consistency=[]
    overall=[]
    classification=[]
    train_accuracy=[]
    steps=[]

    # iterator_unlabel = iter(tar_dataset)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(args.batchSize)

    #training teacher with one epoch 
   
    #this I am doing to get all steps details in epoch
    i=0
    print('Train Mean teacher Model...')

    acc_t=0

    for epoch in range(args.epochs):  
        print(*"*****************")
        print('Start of epoch %d' % (epoch,))
        print(*"*****************")
        #iteration over batches 
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
             with tf.GradientTape() as tape:
         
                # adding instant noise
                iterator_unlabel = iter(tar_dataset)
                x_batch_unlabel = iterator_unlabel.get_next()

                x_train_n,y_train_n= instant_noise(x_batch_train,y_batch_train,x_batch_unlabel,0.2)

                # Run the forward pass of the layer
                logits= student(x_train_n, training= True)  
                logits_acc =  student(x_batch_train, training= False) 

                # TODO:this  metrics also have to right 
                train_metrics(y_batch_train,logits_acc)  

                #Calculating classification cost 
                classification_cost = classification_costs(logits,y_train_n)
                # classification.append(classification_cost)
         
                #calculating consistency cost for unlabelled dataset
                #  iterator_unlabel = iter(tar_dataset)
                #  x_batch_tar = iterator_unlabel.get_next()
                '''experiment and check'''
                #  tar_student = student(x_batch_tar) #x_batch_train try this
                x_train_n1,y_train_n1= instant_noise(x_batch_train,y_batch_train,x_batch_unlabel,0.2)
                tar_teacher = teacher(x_train_n1) #x_batch_train
                #  tar_student= student(x_train_n)
                consistency_cost= Consistency_Cost(tar_teacher,logits) 
                # consistency.append(consistency_cost)

                overall_cost= Overall_Cost(classification_cost, consistency_cost, ratio=0.5)
                # overall.append(overall_cost)
                #  consistency_cost = consistency_cost #this is ratio 
                #adding loss to student model 
             grads= tape.gradient(overall_cost, student.trainable_weights)
             i=i+1
             steps.append(i)
   
             # the value of the variables to minimize the loss.
             optimizer.apply_gradients(zip(grads, student.trainable_weights))
             teacher= ema(student, teacher, alpha=args.alpha)
            #  if step % 10==0:
            #      print('alpha:', alpha)
            #      print('Training loss:- Binary Cross entropy at step %s: %s' % (step, float(classification_cost)))
            #      print("Consistency Cost: %s" % (float(consistency_cost)))
            #      print('Seen so far: %s samples' % ((step + 1) * batch_size))
            #      print("Overall Cost: %s" % (float(overall_cost)))
            #      print("--------------- step, batch %s: %s ---------------------"% (step,((step + 1) * batch_size)))
   
        # if alpha <=0.99:
        #     alpha= alpha+0.005
        train_acc = train_metrics.result()
   
        #appending training accuracy
        train_accuracy.append(train_acc)

        # print('Training acc over epoch: %s' % (float(train_acc)*100,))
            # Reset training metrics at the end of each epoch
        train_metrics.reset_states()
   
        # Run a validation loop at the end of each epoch.

        acc_s = model_evaluation(args,epoch,student, x_test, y_test, name= 'student')

        acc_t = model_evaluation(args,epoch,teacher, x_test, y_test, name= 'teacher')
    cm, test_accuracy, precision, recall, f1_score =Confusion_matrix(student,x_test,y_test,0.51, 'Student model')
    report_writing(args, 'Student', args.batchSize,args.epochs,args.alpha,args.ratio, train_acc.numpy(),test_accuracy, f1_score, precision,recall,'LabelledDataset- Consist_cost btw noise target')
    cm, test_accuracy, precision, recall, f1_score =Confusion_matrix(teacher,x_test,y_test,0.51, 'Teacher model')
    report_writing(args, 'Teacher', args.batchSize,args.epochs,args.alpha,args.ratio, train_acc.numpy(),test_accuracy, f1_score, precision,recall,'Labelled Dataset- Consist_cost btw noise target')
    return
def add_params_mt(parser):
    parser.add_argument('--epoch', type=int)
    return parser