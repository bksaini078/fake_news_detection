# False Article Detection With Weakly Supervised Learning

This contains the code for detecting the false article detection using weakly supervised learning. 

Due to storage constraint, we are not able to load many files such as embedding.model, glove, etc. We have kept them in 
[Google drive](https://drive.google.com/drive/folders/1qbpFysqLRYo_UchAIeTa-hGMHahNsOed?usp=sharing)

## The Data\Input2 is the dataset on which overall comparision have been done and it is default dataset for model.The reports also generated from dataset2

## Starting

Install libraries in requirements.txt:
```console
pip3 install -r requirements.txt
```

## Data Preprocessing for Mean Teacher and Pseduo Label
From the command prompt, change your present working directory to folder where your downloaded code is present. Change the directory to Fakenewdetection folder execute the below command

```console
python Datapreprocessing\Datapreprocessing.py
```

Place the label.csv and unlabel.csv under Data\Preprocessing\Input1 folder. We have created 4 different types of datasets each of them are placed
under Input1, Input2, Input3, Input4 folders respectively.

Summary of the columns present in the data file.

| Column | Description |
|--------|-------------|
|Article | contains true and fake articles |
| Label  | For True news = 1, Fake news = 0 and unlabeled = -1. (numeric value)|


Once the data preprocessing is completed, train data and test data files will be available under Data\Input folder.

Default values 

As the preprocessing is done, the npy files will be available in Data\Input2, which will be input for Mean Teacher and Pseduo Label for training.

For example:-
```console
python Datapreprocessing\Datapreprocessing.py --inputPath C:\fake_news_detection\Data\Preprocessing\Input4\ --outputPath C:\fake_news_detection\Data\Input4\
```

### Parameters required for running Datapreprocessing.py


| Parameter | Description |
|-----------|-------------|
|--inputPath | input path for un-processed data|
|--outputPath | output path for processed data, taken as input for ML models|
|--labeldataname | file name for label data |
|--unlabeldataname | file name for unlabel data |
|--xTrain | npy xTrain file name |
|--xTest | npy xTest file name |
|--yTrain | npy yTrain file name |
|--yTest | npy yTest file name |
|--xUnlabel | npy xUnlabel file name |



### Default values 

| Parameter | Default Value |
|-----------|-------------|
|--inputPath | os.path.abspath(os.getcwd())+"\\Data\\\Preprocessing\\Input2\\ |
|--outputPath | os.path.abspath(os.getcwd())+"\\Data\\Input2\\ |
|--labeldataname | label.csv |
|--unlabeldataname | unlabel.csv |
|--xTrain | xtr_shuffled.npy |
|--xTest | xte_shuffled.npy |
|--yTrain | ytr_shuffled.npy |
|--yTest | yte_shuffled.npy |
|--xUnlabel | xun_shuffled.npy |


We have completed preprocessing and placed all those data files in respective folders such as Data\Input1,Data\Input2,Data\Input3,Data\Input4.
From these folder, preprocessed data can take for further training the meanteacher and pseduolabel models

From the analysis,we found that 

#####  Dataset 1 is over fitting

##### Dataset 2 is tested and working correctly

##### Dataset 3 has less amount of data compare to other dataset

##### Dataset 4 is working fine but not tested completely. 


### Mean Teacher
Mean Teacher model for false article classification.

It will take test and train data from Data\Input2 folder as befault, and once training complete it will place the report and plot in Data\Output1 folder as default
Please Go to folder path where code is available and run like below
Mean Teacher, we have created different method of noises which can be called by parameters 

| Parameter | Description |
|-----------|-------------|
|--meanteacher MT_syn_unlabel_diff | Mean teacher |
|--meanteacher MT_syn_noise_diff | Mean teacher with synonym noise |
|--meanteacher MT_dropout| Mean teacher with dropout |
|--meanteacher MT_syn_dropout | Mean teacher with synonym and dropout |

Once you call these method as parameter, automatically code regarding to this function and train the data in Data\Input1 folder (default)
and the report will be placed in Data\Output1 folder (default) , default folders also changed by parameter passing

```console
python  MeanTeacher\main.py --meanteacher MT_dropout --lr 0.0005 --epochs 5 --batchSize 64 --alpha 0.95
```

other parameters are also available 

| Parameter | Description |
|-----------|-------------|
|--ratio |  |
|--threashold |  |
|--inputPath | Input path to the train model ( /Users/tmp/Testing/) |
|--outputPath | Output path to place report and files ( /Users/tmp/Testing/) |


Default values 

| Parameter | Default Value |
|-----------|-------------|
|--lr | 0.0001 |
|--epochs | 10 |
|--batchSize | 64 |
|--ratio | 0.5 |
|--alpha | 0.97 |
|--dropout | 0.2 |
|--synonym_noise_b1 | 0.5 |
|--synonym_noise_b2 | 0.3 |
|--inputPath | os path where you placed the code in system + \Data\Input2\ |
|--outputPath | os.path.here you placed the code in system + \Data\Output2\ |
|--xTrain |"xtr_shuffled.npy" |
|--xTest | "xte_shuffled.npy"|
|--yTrain | "ytr_shuffled.npy" |
|--yTest | "yte_shuffled.npy" |
|--xUnlabel | "xun_shuffled.npy" |
|--reportIterate | "report_iterate_mean_teacher.csv" |


### Pseduo Label

Pseduo Label model for false article classification.

It will take test and train data from Data\Input2 folder as default, and once training complete it will place the report and plot in Data\Output2 folder as default
Please Go to folder path where code is available and run like below

```console
python PseudoLabel\main.py --inputPath C:\fake_news_detection\Data\Input4\ --epochs 30 --batch_size 32
```

--inputPath  - input path to the train model (Eg: /Users/tmp/Testing/)
--outputPath - output path to place report and files (Eg: /Users/tmp/Testing/)

| Parameter | Default Value |
|-----------|-------------|
|--lr | 0.0001 |
|--Epochs | 14 |
|--batch_size | 64 |
|--Thresh | 0.90 |
|--n | 10 |
|--max_len | 128 |
|--inputPath| os path where you placed the code in system + \Data\Input2\ |
|--outputPath | os.path.here you placed the code in system + \Data\Output2\ |
|--xTrain |"xtr_shuffled.npy" |
|--xTest | "xte_shuffled.npy"|
|--yTrain | "ytr_shuffled.npy" |
|--yTest | "yte_shuffled.npy" |
|--xUnlabel | "xun_shuffled.npy" |

### VAT
#### Data Preprocessing for VAT

VAT used the spacy for the preprocessing the code and its has its own folder strutcure to the run the VAT main code.

Raw data was placed in the VAT\Input1\raw -- which is not processed data (label.csv and unlable.csv files), the processed files will be placed in the temp inside same folder

we have created 4 different types datasets each data sets placed in the different folder like Input1,Input2,Input3,Input4 in the same way. 

Here preprocessing done in four stages

Stage 1 : Here raw (default-VAT\Input1\raw) file wil be processed and placed in processed folder ( default - VAT\Input1\processed)

Example:

```console
python VAT\preprocessing\text_spacy.py --inputPath C:\fake_news_detection\VAT\Input4\raw\ --outputPath C:\fake_news_detection\VAT\Input4\processed\
```
| Parameter | Default Value |
|-----------|-------------|
|--inputPath| \\VAT\\Input1\\raw\\ |
|--labelfilename | label.csv |
|--unlabelfilename | unlabel.csv |
|--outputPath | \\VAT\\Input1\\processed\\ |
|--Remain | label.csv |
|--xUnlabel | "xun_shuffled.npy" |

Stage 2 : split the label data and place it in the processed folder

Example:

```console
python VAT\preprocessing\split_data.py --Path C:\fake_news_detection\VAT\Input2\processed\
```
| Parameter | Default Value |
|-----------|-------------|
|--Path| \\VAT\\Input1\\processed\\ |
|--filename | label.csv |
|--xUnlabel | xun_shuffled.npy |
|--xTrain | x_train.npy |
|--xTest | x_test.npy |
|--yTrain | y_train.npy |
|--yTest | y_test.npy |


Stage 3 : Tokenize the splitted data., here word picket file will be placed in the meta folder

Example:

```console
python VAT\preprocessing\tokenization_vat.py --inputPath C:\fake_news_detection\VAT\Input2\processed\ --outputPath C:\fake_news_detection\VAT\Input2\temp\ --pickelPath C:\fake_news_detection\VAT\Input2\meta\
```

Parameters with default values:

| Parameter | Default Value |
|-----------|-------------|
|--inputPath| \\VAT\\Input1\\processed\\ |
|--outputPath| \\VAT\\Input1\\temp\\ |
|--pickelPath| \\VAT\\Input1\\meta\\ |
|--filename | label.csv |
|--xUnlabel | xun_shuffled.npy |
|--xTrain | x_train.npy |
|--xTest | x_test.npy |
|--yTrain | y_train.npy |
|--yTest | y_test.npy |

Stage 4 : Emedding, for this golve file is need, this file you can download from the site https://nlp.stanford.edu/projects/glove/ and place it in the 
path \\VAT\\ because this files is common to all input files. Once emedding its created and it will place it in the meta folder which is default.

Example :

```console
python VAT\preprocessing\embedding.py --pickelPath C:\fake_news_detection\VAT\Input2\meta\ 
```
Parameters with default values:

| Parameter | Default Value |
|-----------|-------------|
|--pickelPath | \\VAT\\Input1\\meta |
|--glovePath | \\VAT |
|--EMBEDDING_DIM | 300 |

Once all processing completed the final processed file will be placed inside the VAT\Input1\temp\ folder for each data. 
We have completed the preprocessing and those files are already in location.

####  VAT main

Once all files are available in temp directery, then we have to call only the main\vat.py

Example:

```console
python VAT\main\vat.py --n_epochs 30 --batchSize 64 --lr 0.0001 
```

Parameters with default values:

| Parameter | Default Value |
|-----------|-------------|
|--lr | 0.0001 |
|--n_epochs | 10 |
|--batch_size | 32 |
|--emb_dim | 300 |
|--doc_length| 100 |
|--n_splits | 10 |
|--epsilon | 0.01 |
|--act_fn | 'tanh'|
|--model_type | 'VAT' |
|--comment | 'Unknown embeddings 6210/38028' |
|--inputPath | os.path.abspath(os.getcwd())+"\\VAT\\Input1\\" |
|--outputPath | os.path.abspath(os.getcwd())+"\\VAT\\Output1\\" |


## Contributors


Bhupender kumar Saini

Swathi Chidipothu Hare

Mayur Rameshbhai Waghela

Lokesh Sharma

Ravineesh Goud

Ipek Baris
