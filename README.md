# False Article Detection With Weakly Supervised Learning

This contain the code for detecting the false article detection using weakly supervised learning

## Starting


Install libraries in requirements.txt:

`pip3 install -r requirements.txt`


## Data Preprocessing 
From command prompt and Go to folder path where code is available and run the below command
`
python Datapreprocessing\Datapreprocessing.py
`

In the Data\Preprocessing folder place the label.csv and unlable.csv file for data preprocessing

In the data file, these columns and its names are important for data cleaning and creating the test and train data files for models processing. 
once data preprocessing complete it will place the test and train data in Data\Input

`Article` - contain the articles data

`Label` - lables for artilce Real=-1 Fake=- 0 and unlable=-1 and these should be numeric 

Default values 


### Mean Teacher
Mean Teacher model for false article classification.

It will take test and train data from Data\Input folder and once training complete it will place the report and plot in Data\Output folder
Please Go to folder path where code is available and run like below

`python MeanTeacher\main.py --lr 0.0005 --epochs 5 --batchSize 64 --alpha 0.95 
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
--threashold default=0.51
--inputPath default=os path where you placed the code in system + \Data\Input\
--outputPath'os.path.here you placed the code in system + \Data\Output\
--xTrain default="xtr_shuffled.npy"
--xTest default="xte_shuffled.npy"
--yTrain default="ytr_shuffled.npy"
--yTest default="yte_shuffled.npy"
--xUnlabel default="xun_shuffled.npy"


### VAT
TBD

## Acknowledgement
TBD


## Contributors

Bhupender kumar Saini

Swathi Chidipothu Hare

Ravineesh Goud

Mayur Rameshbhai Waghela

Lokesh Sharma

Ipek Baris
