# False Article Detection With Weakly Supervised Learning

This contain the code for detecting the false article detection using weakly supervised learning. 

Becasue of Space issue and constraint, we are not able to load the many files such as embedding.model,glove,etc. The whole set up, we have kept in the Gdrive.Please see the link below.

https://drive.google.com/drive/folders/1qbpFysqLRYo_UchAIeTa-hGMHahNsOed?usp=sharing


## Starting


Install libraries in requirements.txt:

`pip3 install -r requirements.txt`


## Data Preprocessing for Mean Teacher and Pseduo Label
From command prompt and Go to folder path where code is available and run the below command.
Here go to Fakenewdetection folder then excecute the below comment
`
python Datapreprocessing\Datapreprocessing.py
`

In the Data\Preprocessing\Input1 folder place the label.csv and unlable.csv file for data preprocessing and This is raw data, 
because we have created 4 different types datasets each data sets placed in the different folder like Input1,Input2,Input3,Input4

In the data file, these columns and its names are important for data cleaning and creating the test and train data files for models processing. 
once data preprocessing complete it will place the test and train data in Data\Input

`Article` - contain the articles data

`Label` - lables for artilce Real=-1 Fake=- 0 and unlable=-1 and these should be numeric 

Default values 

Once file processed and it will place the npy files in Data\Input1 -- From this location npy files will taken by MeanTeacher and Pseduo label models for training
it not hard code, you can change these location by passing parameters 

Example


`python Datapreprocessing\Datapreprocessing.py --inputPath C:\masters\Master_projects\research_lab\fake_news_detection\Data\Preprocessing\Input4\ --outputPath C:\masters\Master_projects\research_lab\fake_news_detection\Data\Input4\
`
Parameters of Datapreprocessing.py, default parameters have been give, if its needed it can be replaced by passing as parameters. 

--inputPath  - input path for raw data (not processed)

--outputPath -  output Path for processed data, which will intake by further models

--labeldataname - file name for label data

--unlabeldataname - file name for unlabel data

--xTrain - npy xTrain file name

--xTest - npy xTest file name

--yTrain - npy yTrain file name

--yTest - npy yTest file name

--xUnlabel - npy xUnlabel file name


Default values 

--inputPath  - os.path.abspath(os.getcwd())+"\\Data\\\Preprocessing\\Input1\\

--outputPath -  os.path.abspath(os.getcwd())+"\\Data\\Input1\\

--labeldataname - label.csv

--unlabeldataname - unlabel.csv

--xTrain - xtr_shuffled.npy

--xTest - xte_shuffled.npy

--yTrain - ytr_shuffled.npy

--yTest - yte_shuffled.npy

--xUnlabel - xun_shuffled.npy


We have completed preprocessing and placed all those data files in respective folders such as Data\Input1,Data\Input2,Data\Input3,Data\Input4.
From these folder, preprocessed data can take for further training the meanteacher and pseduolabel models
### Mean Teacher
Mean Teacher model for false article classification.

It will take test and train data from Data\Input1 folder as befault, and once training complete it will place the report and plot in Data\Output1 folder as default
Please Go to folder path where code is available and run like below
Mean Teacher, we have created different method of noises which can be called by parameters 

--meanteacher  MT_syn_unlabel_diff (Mean teacher default )

--meanteacher  MT_syn_noise_diff   (Mean teacher with synonym noise)

--meanteacher  MT_dropout  ( Mean teacher with dropout)

--meanteacher  MT_syn_dropout (Mean teacher with synonym and dropout)


Once you call these method as parameter, automatically code regarding to this function and train the data in Data\Input1 folder (default)
and the report will be placed in Data\Output1 folder (default) , default folders also changed by parameter passing
`python  MeanTeacher\main.py --meanteacher MT_dropout --lr 0.0005 --epochs 5 --batchSize 64 --alpha 0.95 
`

other parameters also available 


--ratio

--threashold

--inputPath  - input path to the train model ( /Users/tmp/Testing/)

--outputPath - output path to place report and files ( /Users/tmp/Testing/)


Default values 

--lr default=0.0001
--epochs default=10
--batchSize default=64
--ratio default=0.5
--alpha default=0.97
--dropout default=0.2
--synonym_noise_b1=0.5
--synonym_noise_b2=0.3
--inputPath default=os path where you placed the code in system + \Data\Input1\
--outputPath'os.path.here you placed the code in system + \Data\Output1\
--xTrain default="xtr_shuffled.npy"
--xTest default="xte_shuffled.npy"
--yTrain default="ytr_shuffled.npy"
--yTest default="yte_shuffled.npy"
--xUnlabel default="xun_shuffled.npy"
--reportIterate default="report_iterate_mean_teacher.csv"

### Pseduo Label

Pseduo Label model for false article classification.

It will take test and train data from Data\Input1 folder as befault, and once training complete it will place the report and plot in Data\Output1 folder as default
Please Go to folder path where code is available and run like below

`python PseudoLabel\main.py --inputPath C:\masters\Master_projects\research_lab\fake_news_detection\Data\Input4\ --epochs 30 --batchSize 64 --alpha 0.95 

`
--ratio
--Thresh
--inputPath  - input path to the train model ( /Users/tmp/Testing/)
--outputPath - output path to place report and files ( /Users/tmp/Testing/)
--n

Default values 

--lr default=0.0001

--epochs default=14

--batchSize default=32

--Thresh default=0.90

--inputPath default=os path where you placed the code in system + \Data\Input1\

--outputPath'os.path.here you placed the code in system + \Data\Output1\
--xTrain default="xtr_shuffled.npy"
--xTest default="xte_shuffled.npy"
--yTrain default="ytr_shuffled.npy"
--yTest default="yte_shuffled.npy"
--xUnlabel default="xun_shuffled.npy"

### VAT
#### Data Preprocessing for VAT

VAT used the spacy for the preprocessing the code and its has its own folder strutcure to the run the VAT main code.

Raw data was placed in the VAT\Input1\raw -- which is not processed data (label.csv and unlable.csv files), the processed files will be placed in the temp inside same folder

we have created 4 different types datasets each data sets placed in the different folder like Input1,Input2,Input3,Input4 in the same way. 

Here preprocessing done in four stages

Stage 1 : Here raw (default-VAT\Input1\raw) file wil be processed and placed in processed folder ( default - VAT\Input1\processed)

Example:

`python VAT\preprocessing\text_spacy.py --inputPath C:\masters\Master_projects\research_lab\fake_news_detection\VAT\Input4\raw\ --outputPath C:\masters\Master_projects\research_lab\fake_news_detection\VAT\Input4\processed\

`

Parameters with default values:
--inputPath \\VAT\\Input1\\raw\\
--labelfilename label.csv
--unlabelfilename unlabel.csv
--outputPath \\VAT\\Input1\\processed\\
--Remain lable.csv 
--xUnlabel xun_shuffled.npy 

Stage 2 : split the label data and place it in the processed folder

Example:
 
`python VAT\preprocessing\split_data.py --Path C:\masters\Master_projects\research_lab\fake_news_detection\VAT\Input2\processed\
`
Parameters with default values:
--Path \\VAT\\Input1\\processed\\
--filename lable.csv
--xUnlabel xun_shuffled.npy
--xTrain x_train.npy
--xTest x_test.npy
--yTrain y_train.npy
--yTest y_test.npy

Stage 3 : Tokenize the splitted data., here word picket file will be placed in the meta folder

Example:

`python VAT\preprocessing\tokenization_vat.py --inputPath C:\masters\Master_projects\research_lab\fake_news_detection\VAT\Input2\processed\ --outputPath C:\masters\Master_projects\research_lab\fake_news_detection\VAT\Input2\temp\ --pickelPath C:\masters\Master_projects\research_lab\fake_news_detection\VAT\Input2\meta\
`

Parameters with default values:

--inputPath \\VAT\\Input1\\processed\\
--outputPath \\VAT\\Input1\\temp\\
--pickelPath \\VAT\\Input1\\meta\\
--filename lable.csv
--xUnlabel xun_shuffled.npy
--xTrain x_train.npy
--xTest x_test.npy
--yTrain y_train.npy
--yTest y_test.npy

Stage 4 : Emedding, for this golve file is need, this file you can download from the site https://nlp.stanford.edu/projects/glove/ and place it in the 
path \\VAT\\ because this files is common to all input files. Once emedding its created and it will place it in the meta folder which is default.

Example :
`python VAT\preprocessing\embedding.py --pickelPath C:\masters\Master_projects\research_lab\fake_news_detection\VAT\Input2\meta\ 
`
Parameters with default values:

--pickelPath \\VAT\\Input1\\meta
--glovePath \\VAT
--EMBEDDING_DIM 300 


Once all processing completed the final processed file will be placed inside the VAT\Input1\temp\ folder for each data. 
We have completed the preprocessing and those files are already in location.

####  VAT main

Once all files are available in temp directery, then we have to call only the main\vat.py

Example:
`python VAT\main\vat.py --n_epochs 30 --batchSize 64 --lr 0.0001 
`

Parameters with default values:

--lr, default=0.0001
--n_epochs, default=10
--batch_size, default=32
--emb_dim, default=300
--doc_length, default=100
--n_splits, default=10
--epsilon, default=0.01
--act_fn, default='tanh'
--model_type, default='VAT
--comment, default='Unknown embeddings 6210/38028'
--inputPath, default=os.path.abspath(os.getcwd())+"\\VAT\\Input1\\"
--outputPath, default=os.path.abspath(os.getcwd())+"\\VAT\\Output1\\"



## Contributors

Swathi Chidipothu Hare

Bhupender kumar Saini

Mayur Rameshbhai Waghela

Lokesh Sharma

Ravineesh Goud

Ipek Baris
