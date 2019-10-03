# Code Book
The data set described in this code book was downloaded from two sources. The first source corresponds to the training data set, while the second corresponds to the test data set. 

training data set - https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
test data set - https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


## Data Set Description
Each row in the training data set and test data set contains a total of 160 variables corresponding to the various measurements made using devices such as Jawbone Up, Nike FuelBand, and Fitbit as well as additional information on these measurements. This includes the names of the 6 participants, the date they were taken and so on. The training dataset has 19622 objects while the test data set has 20 variables. 

## Data Transformation
1. Read the data and store them respectively in two separate variables as to designate which is the training, and which is the test dataset.
2. All columns with zero variance are removed in both datasets.
3. All columns with NAs are removed in both datasets.
4. The first 8 columns were removed from the datasets as they appear to be irrelevant in predicting the class of the activity. This, however only applies in the current scenario. Dates, for instance, may affect the predicted class in other cases. 