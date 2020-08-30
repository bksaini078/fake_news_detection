import datetime
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional
from tensorflow.keras.models import Sequential, Model
import argparse

def load_data(path):
    x_train = np.load(path + '/x_train.npy')
    y_train = np.load(path + '/y_train.npy')
    x_test = np.load(path + '/x_test.npy')
    y_test = np.load(path + '/y_test.npy')
    return x_train, y_train, x_test, y_test


def convert_tf_data(data, train_load_batch, test_load_batch):
    x_train, x_test, y_train, y_test = data[0], data[1], data[2], data[3]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(train_load_batch, seed=1, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(train_load_batch)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.shuffle(test_load_batch, seed=1, reshuffle_each_iteration=True)
    test_dataset = test_dataset.batch(test_load_batch)
    return train_dataset, test_dataset


def load_word_index(path):
    """
    Wordnet for the corpus; has the mapping of each word to unique token
    :return: word_index
    """
    word_index = open(path + '/word_index.pickle', 'rb')
    word_index = pickle.load(word_index)
    print('Word Index Pickle load successful\n')
    return word_index


def load_emb_matrix(path):
    """
    Loads the embedding format stored locally as npy format
    :return: embedding_matrix
    """
    embedding_matrix = np.load(path + '/emb_matrix.npy')
    print('Embedding matrix load from local sys successful\n')
    print('Denormalized Shape: ({},{})'.format(np.shape(embedding_matrix)[0], np.shape(embedding_matrix)[1]))
    return embedding_matrix


def embedding(args,seq, matrix):
    # input_dim is +1 because word_index starts from index 1 and embedding matrix from 0
    emb = Embedding(input_dim=len(word_index) + 1,
                    output_dim=args.emb_dim,
                    input_shape=(args.doc_length,),
                    weights=[matrix],
                    trainable=False,
                    mask_zero=True)(seq)
    return emb


def network_architecture(act_fn):
    network = Sequential()
    network.add(Bidirectional(LSTM(units=128)))
    network.add(Dense(units=64, activation=act_fn))
    network.add(Dense(units=32))
    return network


def reset_model(args,act_fn):
    network = network_architecture(act_fn)
    emb_tensor = Input(shape=(args.doc_length, args.emb_dim,))
    logit_tensor = network(emb_tensor)
    output_tensor = Dense(units=2, activation='softmax')(logit_tensor)  # don't add non-linearity here
    model = Model(inputs=emb_tensor, outputs=output_tensor)
    return model


def prec_rec_f1_score(model, x, y_true):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score

    y_hat = model.predict(x)
    y_pred = np.argmax(y_hat, axis=1)

    pr_re_f1score_perclass = precision_recall_fscore_support(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # per class
    precision_true = round(pr_re_f1score_perclass[0][1], 3)
    precision_fake = round(pr_re_f1score_perclass[0][0], 3)
    recall_true = round(pr_re_f1score_perclass[1][1], 3)
    recall_fake = round(pr_re_f1score_perclass[1][0], 3)
    f1score_true = round(pr_re_f1score_perclass[2][1], 3)
    f1score_fake = round(pr_re_f1score_perclass[2][0], 3)
    metrics_name = ['accuracy', 'precision_true', 'precision_fake', 'recall_true', 'recall_fake', 'f1score_true',
                    'f1score_fake']
    metrics_value = [accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake]
    i = 0
    for item in metrics_name:
        print(item + ':', metrics_value[i])
        i += 1
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss = loss_object(y_true, y_hat).numpy()
    print('loss', loss)
    return accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, round(loss,
                                                                                                                 3)


def reports(path, model, K, ratio, lr, batch_size, activation, n_epoch, epsilon, doc_length, embedding_dim,
            train_accuracy, test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1_score_true,
            f1_score_fake, loss, comment):
    time_now = datetime.datetime.now()
    report = pd.DataFrame(
        columns=['Date', 'Model', 'Epsilon', 'Batch Size', 'K Fold', 'Learning Rate', 'Epochs', 'Ratio (test/train)',
                 'Activation',
                 'Train Accuracy', 'Test Accuracy', 'Precision True', 'Precision False', 'Recall True', 'Recall Fake',
                 'F1 Score True',
                 'F1 Score Fake', 'Classification Loss', 'Comment', 'Document Length', 'Embedding dim', ])

    report = report.append({'Date': time_now.strftime("%c"),
                            'Model': model,
                            'Epsilon': epsilon,
                            'Batch Size': batch_size,
                            'K Fold': K,
                            'Learning Rate': lr,
                            'Epochs': n_epoch,
                            'Ratio (test/train)': ratio,
                            'Activation': activation,
                            'Train Accuracy': train_accuracy,
                            'Test Accuracy': test_accuracy,
                            'Precision True': precision_true,
                            'Precision False': precision_fake,
                            'Recall True': recall_true,
                            'Recall Fake': recall_fake,
                            'F1 Score True': f1_score_true,
                            'F1 Score Fake': f1_score_fake,
                            'Classification Loss': loss,
                            'Comment': comment,
                            'Document Length': doc_length,
                            'Embedding dim': embedding_dim, }, ignore_index=True)

    report_path = Path(path + '/Report_VAT_Spacy.csv')

    if report_path.exists():
        report.to_csv(report_path, mode='a', header=False, index=False)
        print('New results logged in existing report file')
    else:
        report.to_csv(report_path, mode='w', header=True, index=False)
        print('New Report created')

def supervised(lr=0.002, n_epochs=4, model=None, batch_size=32, train_dataset=None, test_dataset=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                         amsgrad=False, name='Adam')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model_supervised = model
    dataset_batch = 1
    last_accuracy_list = []
    model_supervised.compile(optimizer=optimizer, loss=loss_object, metrics='accuracy')

    for features, labels in train_dataset:
        print('Training on dataset batch: ', dataset_batch, '_' * 80)
        dataset_batch += 1
        clean_emb = embedding(args,features, embedding_matrix)
        print('Training dataset batch size: ', clean_emb.shape, '\n')
        train_history = model_supervised.fit(x=clean_emb, y=labels, validation_split=0.1,
                                             batch_size=batch_size, epochs=n_epochs)
        accuracy = train_history.history['accuracy'][len(train_history.epoch) - 1]
        last_accuracy_list.append(accuracy)

    train_accuracy = np.mean(last_accuracy_list)
    print('Mean training accuracy over all batches: ', train_accuracy, '\n')
    print('Test Dataset Evaluation:', '_' * 80)

    test_feature, test_label = next(iter(test_dataset))
    test_emb = embedding(args,test_feature, embedding_matrix)
    print('Test dataset batch size: ', test_emb.shape, '\n')
    model_supervised.evaluate(test_emb, test_label)

    test_accuracy, precision_true, precision_fake, recall_true, recall_fake,\
    f1score_true, f1score_fake, loss = prec_rec_f1_score(model_supervised, test_emb, test_label)
    print('*' * 55, 'Done', '*' * 55)

    return train_accuracy, test_accuracy, precision_true, precision_fake, recall_true, recall_fake,\
           f1score_true, f1score_fake, loss, model_supervised


def compute_kld(p_logit, q_logit):
    p = tf.nn.softmax(p_logit)
    q = tf.nn.softmax(q_logit)
    kl_loss_object = tf.keras.losses.KLDivergence()
    kl_score = kl_loss_object(p, q)
    return kl_score


def vat_carrier(p_logit, clean_emb, epsilon, network):
    noise_emb = tf.random.uniform(shape=tf.shape(clean_emb), minval=- epsilon, maxval=epsilon)
    noise_emb = tf.add(clean_emb, noise_emb)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(noise_emb)
        p_logit_r = network(noise_emb)
        kl_score = compute_kld(p_logit, p_logit_r)
    grads = tape.gradient(kl_score, noise_emb)
    p_logit = tf.stop_gradient(p_logit)
    grads = tf.stop_gradient(grads)

    norm_ball_2 = tf.math.l2_normalize(grads, axis=0)
    r_vadv = (grads / norm_ball_2) * -1
    r_vadv_emb = tf.add(clean_emb, r_vadv)
    q_logit = network(r_vadv_emb)
    vat_loss = compute_kld(p_logit, q_logit)
    return vat_loss


def vat(lr=0.002, n_epochs=4, model=None, act_fn='tanh', epsilon=0.01, batch_size=32, train_dataset=None,
        test_dataset=None):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                         amsgrad=False, name='Adam')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model_vat = model
    network = network_architecture(act_fn)
    dataset_batch = 1
    last_accuracy_list = []
    model_vat.compile(optimizer=optimizer, loss=loss_object, metrics='accuracy')

    for features, labels in train_dataset:
        print('Training on dataset batch: ', dataset_batch, '_' * 80)
        dataset_batch += 1
        clean_emb = embedding(args,features, embedding_matrix)
        print('Training dataset batch size: ', clean_emb.shape, '\n')
        print('Calculating Vat Loss and logits for embeddings...')
        p_logit = network(clean_emb)
        vat_loss = vat_carrier(p_logit, clean_emb, epsilon, network)

        model_vat.add_loss(lambda: vat_loss)
        print('Vat loss:', vat_loss)
        train_history = model_vat.fit(x=clean_emb, y=labels, validation_split=0.1,
                                      batch_size=batch_size, epochs=n_epochs)
        accuracy = train_history.history['accuracy'][len(train_history.epoch) - 1]
        last_accuracy_list.append(accuracy)

    train_accuracy = np.mean(last_accuracy_list)
    print('Mean training accuracy over all batches: ', train_accuracy, '\n')
    print('Test Dataset Evaluation:', '_' * 80)

    test_feature, test_label = next(iter(test_dataset))
    test_emb = embedding(args,test_feature, embedding_matrix)
    print('Test dataset batch size: ', test_emb.shape, '\n')
    model_vat.evaluate(test_emb, test_label)
    print('\n')

    test_accuracy, precision_true, precision_fake, recall_true, recall_fake, \
    f1score_true, f1score_fake, loss = prec_rec_f1_score(model_vat, test_emb, test_label)
    print('*' * 55, 'Done', '*' * 55)

    return train_accuracy, test_accuracy, precision_true, precision_fake, recall_true, recall_fake, \
           f1score_true, f1score_fake, loss, model_vat


if __name__ == '__main__':
    
    # parameters from arugument parser 
    parser = argparse.ArgumentParser()

    
    # for VAT
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--n_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--emb_dim', default=300, type=int)
    parser.add_argument('--doc_length', default=100, type=int)
    parser.add_argument('--n_splits', default=10, type=int)
    parser.add_argument('--epsilon', default=0.01, type=float)
    parser.add_argument('--act_fn', default='tanh', type=str)
    parser.add_argument('--model_type', default='VAT', type=str)
    parser.add_argument('--comment', default='Unknown embeddings 6210/38028', type=str)
    parser.add_argument('--inputPath',type=str, default=os.path.abspath(os.getcwd())+"\\VAT\\Input1\\")
    parser.add_argument('--outputPath',type=str, default=os.path.abspath(os.getcwd())+"\\VAT\\Output1\\")

    args = parser.parse_args()
    # cwd = os.getcwd()
    # print('Working Directory: ', cwd)
    # path = cwd + '/Fake News Spacy/data'
    # print('Changed directory: ', path)

    #____________________________________________
    word_index = load_word_index(args.inputPath + '/meta')
    embedding_matrix = load_emb_matrix(args.inputPath + '/meta')
    x_train, y_train, x_test, y_test = load_data(args.inputPath + '/temp')
    #____________________________________________

    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

    print('Dimensions of splitted datasets: ')
    print(' Training: {}\n Testing: {}\n'.format((x_train.shape, y_train.shape),
                                                 (x_test.shape, y_test.shape)))

    # #____________________________________________
    # lr 0.001
    # n_epochs = 10
    # emb_dim = 300
    # doc_length = 100
    # batch_size = 32
    # n_splits = 10
    # epsilon = 0.01
    # act_fn = 'tanh'
    # #model_type = 'Supervised'
    # model_type = 'VAT'
    # comment = 'Unknown embeddings 6210/38028'
    # #____________________________________________
    model_name = 'None'
    ratio = str(len(x_test)) + '/' + str(len(x_train))


    K = 1
    kf = KFold(args.n_splits)
    tr_a_l, te_a_l, p_t_l, p_f_l, r_t_l, r_f_l, f1_t_l, f1_f_l, cl_l = [], [], [], [], [], [], [], [], []

    for train_ind, test_ind in kf.split(X):
        model = reset_model(args,args.act_fn)
        print('K fold counter: ', K)
        K += 1
        x_train, x_test = X[train_ind], X[test_ind]
        y_train, y_test = Y[train_ind], Y[test_ind]
        data = [x_train, x_test, y_train, y_test]
        train_load_batch = len(x_train)
        test_load_batch = len(x_test)
        ratio = str(len(x_test)) + '/' + str(len(x_train))

        if args.model_type == 'Supervised':
            print('Running Supervised model...')
            model_name, epsilon = 'Supervised', 'None'
            train_dataset, test_dataset = convert_tf_data(data, train_load_batch,
                                                          test_load_batch)

            train_accuracy, test_accuracy, precision_true, precision_fake, recall_true, recall_fake, \
            f1_score_true, f1_score_fake, classification_loss, model = supervised(args.lr, args.n_epochs,
                                                                                  model,
                                                                                  args.batch_size,
                                                                                  train_dataset,
                                                                                  test_dataset)

        elif args.model_type == 'VAT':
            print('Running Virtual Adversarial Training...')
            model_name = 'VAT_Glove'
            train_dataset, test_dataset = convert_tf_data(data, train_load_batch,
                                                          test_load_batch)
            train_accuracy, test_accuracy, precision_true, precision_fake, recall_true, recall_fake, \
            f1_score_true, f1_score_fake, classification_loss, model = vat(args.lr, args.n_epochs, model,
                                                                           args.act_fn, args.epsilon,
                                                                           args.batch_size,
                                                                           train_dataset,
                                                                           test_dataset)
        else:
            print('Wrong Model Name')
            break
        tr_a_l.append(train_accuracy)
        te_a_l.append(test_accuracy)
        p_t_l.append(precision_true)
        p_f_l.append(precision_fake)
        r_t_l.append(recall_true)
        r_f_l.append(recall_fake)
        f1_t_l.append(f1_score_true)
        f1_f_l.append(f1_score_fake)
        cl_l.append(classification_loss)

    train_accuracy = np.mean(tr_a_l)
    test_accuracy = np.mean(te_a_l)
    precision_true = np.mean(p_t_l)
    precision_fake = np.mean(p_f_l)
    recall_true = np.mean(r_t_l)
    recall_fake = np.mean(r_f_l)
    f1_score_true = np.mean(f1_t_l)
    f1_score_fake = np.mean(f1_f_l)
    classification_loss = np.mean(cl_l)

    reports(args.outputPath, model_name, args.n_splits, ratio, args.lr, args.batch_size, args.act_fn, args.n_epochs, args.epsilon,
            args.doc_length, args.emb_dim,
            train_accuracy, test_accuracy, precision_true, precision_fake, recall_true,
            recall_fake,
            f1_score_true, f1_score_fake, classification_loss, args.comment)

