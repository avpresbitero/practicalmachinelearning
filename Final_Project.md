Practical Machine Learning Final Project
================
Alva Presbitero
9/27/2019

## Practical Machine Learning Final Project

**Background**

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>
(see the section on the Weight Lifting Exercise Dataset).

**Data**

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.
If you use the document you create for this class for any purpose please
cite them as they have been very generous in allowing their data to be
used for this kind of assignment.

## Reading the Data

Import necessary libraries.

``` r
library(RCurl)
```

    ## Loading required package: bitops

``` r
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

Then we import and read the data. Since the data given is a url, we can
easily import the data like so. We read the csv file using the read.csv
command. Then we store the data in the trainng and testing variables as
dataframes.

``` r
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url_train)
testing <- read.csv(url_test)
```

For sanity check, we look at the dimensions of the dataframes for the
training and test data sets that we have just read.

``` r
dim(training); dim(testing);
```

    ## [1] 19622   160

    ## [1]  20 160

``` r
View(training); View(testing);
```

As well as the distributions of the classes being predicted. I plot them
below:

``` r
plot(training$classe, col="yellow", main="Frequency Distributions of Classes", xlab="classe levels", ylab="Frequency")
```

![](Final_Project_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

## Cleaning the Data

Next we remove all columns with zero variance, meaning those columns
that contain only a single value. These features will turn out to be
useless in training as they do not carry information for prediction.

``` r
nzv_cols <- nearZeroVar(training)
if(length(nzv_cols) > 0) training <- training[, -nzv_cols]
if(length(nzv_cols) > 0) testing <- testing[, -nzv_cols]
```

We also remove all columns containing NAs.

``` r
na_names_cols <- names(training)[sapply(training, anyNA)]
testing <- testing[ , -which(names(testing) %in% na_names_cols)]
training <- training[ , -which(names(training) %in% na_names_cols)]
```

Finally, we pinpoint columns that obviously do not relate to the class
predictions. These are the first 8 columns in the dataset which I have
identified as the following
columns:

``` r
delete_cols <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
training <- training[ , -which(names(training) %in% delete_cols)]
testing <- testing[ , -which(names(testing) %in% delete_cols)]
```

For another sanity check, I look into the dimensions of my cleaned
training and test set.

``` r
dim(training); dim(testing)
```

    ## [1] 19622    53

    ## [1] 20 53

## Data Splitting

I then partition my training data set into 60% training and 40%
validation.

``` r
set.seed(100)
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
training_set <- training[inTrain, ]
validation_set <- training[-inTrain, ]
```

## Training and Validating the Ensemble Model

For this project, I will be using an ensemble model called stacking. I
will be using three commonly used machine learning models such as
gradient boosting method, random forest, and linear discriminant
analysis.

I establish each of my individual models by training each model to the
training set. I save these models as an rds file so that I would not
have to call them again, just in case.

``` r
set.seed(100)
# Random Forest
mod_rf <- train(classe ~., data = training_set, method = "rf")
saveRDS(mod_rf, "mod_rf.rds")
mod_rf <- readRDS("mod_rf.rds")

# Gradient Boost Method
mod_gbm <- train(classe ~., data = training_set, method = "gbm")
```

    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1248
    ##      2        1.5250             nan     0.1000    0.0886
    ##      3        1.4668             nan     0.1000    0.0645
    ##      4        1.4235             nan     0.1000    0.0550
    ##      5        1.3878             nan     0.1000    0.0419
    ##      6        1.3592             nan     0.1000    0.0450
    ##      7        1.3300             nan     0.1000    0.0363
    ##      8        1.3069             nan     0.1000    0.0384
    ##      9        1.2823             nan     0.1000    0.0267
    ##     10        1.2642             nan     0.1000    0.0301
    ##     20        1.1080             nan     0.1000    0.0156
    ##     40        0.9358             nan     0.1000    0.0072
    ##     60        0.8305             nan     0.1000    0.0052
    ##     80        0.7489             nan     0.1000    0.0044
    ##    100        0.6849             nan     0.1000    0.0047
    ##    120        0.6339             nan     0.1000    0.0030
    ##    140        0.5883             nan     0.1000    0.0032
    ##    150        0.5685             nan     0.1000    0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1810
    ##      2        1.4913             nan     0.1000    0.1238
    ##      3        1.4103             nan     0.1000    0.1031
    ##      4        1.3449             nan     0.1000    0.0842
    ##      5        1.2924             nan     0.1000    0.0719
    ##      6        1.2457             nan     0.1000    0.0598
    ##      7        1.2072             nan     0.1000    0.0638
    ##      8        1.1679             nan     0.1000    0.0511
    ##      9        1.1349             nan     0.1000    0.0395
    ##     10        1.1077             nan     0.1000    0.0429
    ##     20        0.9040             nan     0.1000    0.0251
    ##     40        0.6899             nan     0.1000    0.0173
    ##     60        0.5545             nan     0.1000    0.0063
    ##     80        0.4672             nan     0.1000    0.0087
    ##    100        0.3987             nan     0.1000    0.0048
    ##    120        0.3459             nan     0.1000    0.0026
    ##    140        0.3041             nan     0.1000    0.0016
    ##    150        0.2875             nan     0.1000    0.0029
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2349
    ##      2        1.4607             nan     0.1000    0.1585
    ##      3        1.3589             nan     0.1000    0.1202
    ##      4        1.2826             nan     0.1000    0.1066
    ##      5        1.2156             nan     0.1000    0.0958
    ##      6        1.1558             nan     0.1000    0.0744
    ##      7        1.1099             nan     0.1000    0.0604
    ##      8        1.0711             nan     0.1000    0.0648
    ##      9        1.0304             nan     0.1000    0.0579
    ##     10        0.9943             nan     0.1000    0.0460
    ##     20        0.7601             nan     0.1000    0.0301
    ##     40        0.5246             nan     0.1000    0.0142
    ##     60        0.4001             nan     0.1000    0.0058
    ##     80        0.3183             nan     0.1000    0.0040
    ##    100        0.2595             nan     0.1000    0.0027
    ##    120        0.2138             nan     0.1000    0.0033
    ##    140        0.1810             nan     0.1000    0.0014
    ##    150        0.1673             nan     0.1000    0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1299
    ##      2        1.5249             nan     0.1000    0.0848
    ##      3        1.4687             nan     0.1000    0.0650
    ##      4        1.4262             nan     0.1000    0.0519
    ##      5        1.3920             nan     0.1000    0.0486
    ##      6        1.3593             nan     0.1000    0.0456
    ##      7        1.3306             nan     0.1000    0.0364
    ##      8        1.3069             nan     0.1000    0.0321
    ##      9        1.2857             nan     0.1000    0.0365
    ##     10        1.2615             nan     0.1000    0.0318
    ##     20        1.1047             nan     0.1000    0.0191
    ##     40        0.9315             nan     0.1000    0.0104
    ##     60        0.8196             nan     0.1000    0.0076
    ##     80        0.7391             nan     0.1000    0.0059
    ##    100        0.6753             nan     0.1000    0.0038
    ##    120        0.6241             nan     0.1000    0.0033
    ##    140        0.5801             nan     0.1000    0.0020
    ##    150        0.5616             nan     0.1000    0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1845
    ##      2        1.4904             nan     0.1000    0.1323
    ##      3        1.4058             nan     0.1000    0.1077
    ##      4        1.3381             nan     0.1000    0.0797
    ##      5        1.2863             nan     0.1000    0.0744
    ##      6        1.2380             nan     0.1000    0.0630
    ##      7        1.1974             nan     0.1000    0.0640
    ##      8        1.1575             nan     0.1000    0.0497
    ##      9        1.1260             nan     0.1000    0.0480
    ##     10        1.0960             nan     0.1000    0.0450
    ##     20        0.8929             nan     0.1000    0.0198
    ##     40        0.6772             nan     0.1000    0.0121
    ##     60        0.5522             nan     0.1000    0.0087
    ##     80        0.4612             nan     0.1000    0.0050
    ##    100        0.3961             nan     0.1000    0.0043
    ##    120        0.3387             nan     0.1000    0.0022
    ##    140        0.2974             nan     0.1000    0.0022
    ##    150        0.2779             nan     0.1000    0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2374
    ##      2        1.4602             nan     0.1000    0.1617
    ##      3        1.3565             nan     0.1000    0.1263
    ##      4        1.2765             nan     0.1000    0.1069
    ##      5        1.2094             nan     0.1000    0.0848
    ##      6        1.1552             nan     0.1000    0.0768
    ##      7        1.1063             nan     0.1000    0.0736
    ##      8        1.0600             nan     0.1000    0.0659
    ##      9        1.0189             nan     0.1000    0.0529
    ##     10        0.9856             nan     0.1000    0.0571
    ##     20        0.7506             nan     0.1000    0.0226
    ##     40        0.5157             nan     0.1000    0.0124
    ##     60        0.3940             nan     0.1000    0.0069
    ##     80        0.3117             nan     0.1000    0.0033
    ##    100        0.2536             nan     0.1000    0.0026
    ##    120        0.2116             nan     0.1000    0.0019
    ##    140        0.1775             nan     0.1000    0.0016
    ##    150        0.1649             nan     0.1000    0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1294
    ##      2        1.5221             nan     0.1000    0.0850
    ##      3        1.4647             nan     0.1000    0.0693
    ##      4        1.4191             nan     0.1000    0.0565
    ##      5        1.3824             nan     0.1000    0.0520
    ##      6        1.3494             nan     0.1000    0.0464
    ##      7        1.3196             nan     0.1000    0.0342
    ##      8        1.2968             nan     0.1000    0.0321
    ##      9        1.2758             nan     0.1000    0.0319
    ##     10        1.2551             nan     0.1000    0.0309
    ##     20        1.0967             nan     0.1000    0.0164
    ##     40        0.9258             nan     0.1000    0.0099
    ##     60        0.8177             nan     0.1000    0.0055
    ##     80        0.7348             nan     0.1000    0.0038
    ##    100        0.6742             nan     0.1000    0.0030
    ##    120        0.6230             nan     0.1000    0.0039
    ##    140        0.5772             nan     0.1000    0.0032
    ##    150        0.5565             nan     0.1000    0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1886
    ##      2        1.4881             nan     0.1000    0.1267
    ##      3        1.4050             nan     0.1000    0.1050
    ##      4        1.3378             nan     0.1000    0.0789
    ##      5        1.2863             nan     0.1000    0.0836
    ##      6        1.2345             nan     0.1000    0.0647
    ##      7        1.1931             nan     0.1000    0.0618
    ##      8        1.1539             nan     0.1000    0.0533
    ##      9        1.1209             nan     0.1000    0.0496
    ##     10        1.0894             nan     0.1000    0.0363
    ##     20        0.8865             nan     0.1000    0.0237
    ##     40        0.6740             nan     0.1000    0.0116
    ##     60        0.5409             nan     0.1000    0.0075
    ##     80        0.4537             nan     0.1000    0.0048
    ##    100        0.3870             nan     0.1000    0.0031
    ##    120        0.3359             nan     0.1000    0.0029
    ##    140        0.2947             nan     0.1000    0.0028
    ##    150        0.2760             nan     0.1000    0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2404
    ##      2        1.4582             nan     0.1000    0.1677
    ##      3        1.3525             nan     0.1000    0.1236
    ##      4        1.2742             nan     0.1000    0.1162
    ##      5        1.2031             nan     0.1000    0.0823
    ##      6        1.1492             nan     0.1000    0.0759
    ##      7        1.1014             nan     0.1000    0.0723
    ##      8        1.0569             nan     0.1000    0.0654
    ##      9        1.0154             nan     0.1000    0.0488
    ##     10        0.9838             nan     0.1000    0.0594
    ##     20        0.7503             nan     0.1000    0.0255
    ##     40        0.5260             nan     0.1000    0.0125
    ##     60        0.3964             nan     0.1000    0.0079
    ##     80        0.3128             nan     0.1000    0.0042
    ##    100        0.2545             nan     0.1000    0.0036
    ##    120        0.2125             nan     0.1000    0.0023
    ##    140        0.1779             nan     0.1000    0.0015
    ##    150        0.1633             nan     0.1000    0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1364
    ##      2        1.5201             nan     0.1000    0.0908
    ##      3        1.4598             nan     0.1000    0.0682
    ##      4        1.4140             nan     0.1000    0.0574
    ##      5        1.3764             nan     0.1000    0.0459
    ##      6        1.3468             nan     0.1000    0.0457
    ##      7        1.3176             nan     0.1000    0.0419
    ##      8        1.2911             nan     0.1000    0.0351
    ##      9        1.2685             nan     0.1000    0.0301
    ##     10        1.2489             nan     0.1000    0.0336
    ##     20        1.0897             nan     0.1000    0.0171
    ##     40        0.9183             nan     0.1000    0.0102
    ##     60        0.8114             nan     0.1000    0.0055
    ##     80        0.7333             nan     0.1000    0.0042
    ##    100        0.6709             nan     0.1000    0.0045
    ##    120        0.6179             nan     0.1000    0.0038
    ##    140        0.5724             nan     0.1000    0.0030
    ##    150        0.5520             nan     0.1000    0.0024
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1908
    ##      2        1.4880             nan     0.1000    0.1362
    ##      3        1.4026             nan     0.1000    0.1123
    ##      4        1.3326             nan     0.1000    0.0864
    ##      5        1.2771             nan     0.1000    0.0761
    ##      6        1.2293             nan     0.1000    0.0648
    ##      7        1.1876             nan     0.1000    0.0560
    ##      8        1.1513             nan     0.1000    0.0574
    ##      9        1.1157             nan     0.1000    0.0461
    ##     10        1.0862             nan     0.1000    0.0455
    ##     20        0.8743             nan     0.1000    0.0218
    ##     40        0.6644             nan     0.1000    0.0129
    ##     60        0.5403             nan     0.1000    0.0071
    ##     80        0.4514             nan     0.1000    0.0040
    ##    100        0.3861             nan     0.1000    0.0051
    ##    120        0.3332             nan     0.1000    0.0036
    ##    140        0.2909             nan     0.1000    0.0025
    ##    150        0.2725             nan     0.1000    0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2394
    ##      2        1.4554             nan     0.1000    0.1717
    ##      3        1.3456             nan     0.1000    0.1314
    ##      4        1.2633             nan     0.1000    0.1021
    ##      5        1.1982             nan     0.1000    0.0895
    ##      6        1.1417             nan     0.1000    0.0781
    ##      7        1.0928             nan     0.1000    0.0629
    ##      8        1.0525             nan     0.1000    0.0602
    ##      9        1.0144             nan     0.1000    0.0623
    ##     10        0.9758             nan     0.1000    0.0536
    ##     20        0.7455             nan     0.1000    0.0201
    ##     40        0.5185             nan     0.1000    0.0159
    ##     60        0.3912             nan     0.1000    0.0091
    ##     80        0.3092             nan     0.1000    0.0030
    ##    100        0.2534             nan     0.1000    0.0042
    ##    120        0.2113             nan     0.1000    0.0017
    ##    140        0.1777             nan     0.1000    0.0022
    ##    150        0.1637             nan     0.1000    0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1319
    ##      2        1.5215             nan     0.1000    0.0912
    ##      3        1.4631             nan     0.1000    0.0674
    ##      4        1.4180             nan     0.1000    0.0552
    ##      5        1.3822             nan     0.1000    0.0452
    ##      6        1.3531             nan     0.1000    0.0462
    ##      7        1.3232             nan     0.1000    0.0363
    ##      8        1.2987             nan     0.1000    0.0330
    ##      9        1.2776             nan     0.1000    0.0336
    ##     10        1.2538             nan     0.1000    0.0294
    ##     20        1.0989             nan     0.1000    0.0166
    ##     40        0.9288             nan     0.1000    0.0103
    ##     60        0.8232             nan     0.1000    0.0062
    ##     80        0.7422             nan     0.1000    0.0046
    ##    100        0.6779             nan     0.1000    0.0046
    ##    120        0.6253             nan     0.1000    0.0024
    ##    140        0.5816             nan     0.1000    0.0020
    ##    150        0.5624             nan     0.1000    0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1895
    ##      2        1.4865             nan     0.1000    0.1324
    ##      3        1.4027             nan     0.1000    0.1075
    ##      4        1.3350             nan     0.1000    0.0829
    ##      5        1.2815             nan     0.1000    0.0790
    ##      6        1.2317             nan     0.1000    0.0591
    ##      7        1.1935             nan     0.1000    0.0652
    ##      8        1.1530             nan     0.1000    0.0580
    ##      9        1.1168             nan     0.1000    0.0441
    ##     10        1.0887             nan     0.1000    0.0472
    ##     20        0.8846             nan     0.1000    0.0216
    ##     40        0.6736             nan     0.1000    0.0093
    ##     60        0.5417             nan     0.1000    0.0064
    ##     80        0.4550             nan     0.1000    0.0057
    ##    100        0.3881             nan     0.1000    0.0041
    ##    120        0.3338             nan     0.1000    0.0030
    ##    140        0.2914             nan     0.1000    0.0033
    ##    150        0.2719             nan     0.1000    0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2360
    ##      2        1.4597             nan     0.1000    0.1616
    ##      3        1.3559             nan     0.1000    0.1326
    ##      4        1.2721             nan     0.1000    0.1078
    ##      5        1.2044             nan     0.1000    0.0929
    ##      6        1.1469             nan     0.1000    0.0767
    ##      7        1.0977             nan     0.1000    0.0590
    ##      8        1.0588             nan     0.1000    0.0658
    ##      9        1.0181             nan     0.1000    0.0520
    ##     10        0.9848             nan     0.1000    0.0460
    ##     20        0.7488             nan     0.1000    0.0292
    ##     40        0.5171             nan     0.1000    0.0142
    ##     60        0.3929             nan     0.1000    0.0053
    ##     80        0.3095             nan     0.1000    0.0043
    ##    100        0.2520             nan     0.1000    0.0039
    ##    120        0.2087             nan     0.1000    0.0016
    ##    140        0.1761             nan     0.1000    0.0017
    ##    150        0.1631             nan     0.1000    0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1272
    ##      2        1.5250             nan     0.1000    0.0872
    ##      3        1.4665             nan     0.1000    0.0640
    ##      4        1.4237             nan     0.1000    0.0536
    ##      5        1.3884             nan     0.1000    0.0498
    ##      6        1.3567             nan     0.1000    0.0405
    ##      7        1.3303             nan     0.1000    0.0423
    ##      8        1.3040             nan     0.1000    0.0341
    ##      9        1.2820             nan     0.1000    0.0314
    ##     10        1.2618             nan     0.1000    0.0335
    ##     20        1.1064             nan     0.1000    0.0176
    ##     40        0.9326             nan     0.1000    0.0081
    ##     60        0.8224             nan     0.1000    0.0072
    ##     80        0.7430             nan     0.1000    0.0042
    ##    100        0.6772             nan     0.1000    0.0030
    ##    120        0.6270             nan     0.1000    0.0030
    ##    140        0.5822             nan     0.1000    0.0025
    ##    150        0.5626             nan     0.1000    0.0017
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1852
    ##      2        1.4886             nan     0.1000    0.1332
    ##      3        1.4053             nan     0.1000    0.1018
    ##      4        1.3398             nan     0.1000    0.0844
    ##      5        1.2856             nan     0.1000    0.0695
    ##      6        1.2416             nan     0.1000    0.0679
    ##      7        1.1998             nan     0.1000    0.0607
    ##      8        1.1616             nan     0.1000    0.0521
    ##      9        1.1284             nan     0.1000    0.0462
    ##     10        1.0997             nan     0.1000    0.0488
    ##     20        0.8943             nan     0.1000    0.0195
    ##     40        0.6749             nan     0.1000    0.0120
    ##     60        0.5464             nan     0.1000    0.0089
    ##     80        0.4537             nan     0.1000    0.0061
    ##    100        0.3841             nan     0.1000    0.0042
    ##    120        0.3282             nan     0.1000    0.0021
    ##    140        0.2876             nan     0.1000    0.0033
    ##    150        0.2713             nan     0.1000    0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2356
    ##      2        1.4589             nan     0.1000    0.1637
    ##      3        1.3555             nan     0.1000    0.1273
    ##      4        1.2757             nan     0.1000    0.1014
    ##      5        1.2127             nan     0.1000    0.0857
    ##      6        1.1580             nan     0.1000    0.0804
    ##      7        1.1061             nan     0.1000    0.0686
    ##      8        1.0624             nan     0.1000    0.0576
    ##      9        1.0252             nan     0.1000    0.0522
    ##     10        0.9921             nan     0.1000    0.0535
    ##     20        0.7536             nan     0.1000    0.0237
    ##     40        0.5227             nan     0.1000    0.0136
    ##     60        0.3909             nan     0.1000    0.0074
    ##     80        0.3110             nan     0.1000    0.0049
    ##    100        0.2506             nan     0.1000    0.0022
    ##    120        0.2083             nan     0.1000    0.0024
    ##    140        0.1759             nan     0.1000    0.0013
    ##    150        0.1629             nan     0.1000    0.0008
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1273
    ##      2        1.5225             nan     0.1000    0.0870
    ##      3        1.4654             nan     0.1000    0.0672
    ##      4        1.4212             nan     0.1000    0.0527
    ##      5        1.3861             nan     0.1000    0.0499
    ##      6        1.3534             nan     0.1000    0.0410
    ##      7        1.3272             nan     0.1000    0.0361
    ##      8        1.3036             nan     0.1000    0.0352
    ##      9        1.2810             nan     0.1000    0.0333
    ##     10        1.2596             nan     0.1000    0.0325
    ##     20        1.1033             nan     0.1000    0.0162
    ##     40        0.9318             nan     0.1000    0.0106
    ##     60        0.8226             nan     0.1000    0.0072
    ##     80        0.7420             nan     0.1000    0.0046
    ##    100        0.6781             nan     0.1000    0.0052
    ##    120        0.6252             nan     0.1000    0.0025
    ##    140        0.5813             nan     0.1000    0.0029
    ##    150        0.5620             nan     0.1000    0.0030
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1824
    ##      2        1.4895             nan     0.1000    0.1274
    ##      3        1.4055             nan     0.1000    0.1050
    ##      4        1.3387             nan     0.1000    0.0785
    ##      5        1.2868             nan     0.1000    0.0845
    ##      6        1.2343             nan     0.1000    0.0646
    ##      7        1.1940             nan     0.1000    0.0534
    ##      8        1.1591             nan     0.1000    0.0542
    ##      9        1.1250             nan     0.1000    0.0437
    ##     10        1.0964             nan     0.1000    0.0410
    ##     20        0.8927             nan     0.1000    0.0229
    ##     40        0.6835             nan     0.1000    0.0150
    ##     60        0.5545             nan     0.1000    0.0070
    ##     80        0.4611             nan     0.1000    0.0065
    ##    100        0.3922             nan     0.1000    0.0052
    ##    120        0.3402             nan     0.1000    0.0020
    ##    140        0.2972             nan     0.1000    0.0017
    ##    150        0.2807             nan     0.1000    0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2303
    ##      2        1.4608             nan     0.1000    0.1614
    ##      3        1.3563             nan     0.1000    0.1239
    ##      4        1.2778             nan     0.1000    0.1040
    ##      5        1.2133             nan     0.1000    0.0906
    ##      6        1.1551             nan     0.1000    0.0769
    ##      7        1.1070             nan     0.1000    0.0639
    ##      8        1.0657             nan     0.1000    0.0559
    ##      9        1.0301             nan     0.1000    0.0635
    ##     10        0.9907             nan     0.1000    0.0470
    ##     20        0.7581             nan     0.1000    0.0233
    ##     40        0.5323             nan     0.1000    0.0123
    ##     60        0.4047             nan     0.1000    0.0069
    ##     80        0.3193             nan     0.1000    0.0049
    ##    100        0.2637             nan     0.1000    0.0022
    ##    120        0.2198             nan     0.1000    0.0039
    ##    140        0.1875             nan     0.1000    0.0013
    ##    150        0.1737             nan     0.1000    0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1275
    ##      2        1.5251             nan     0.1000    0.0863
    ##      3        1.4691             nan     0.1000    0.0663
    ##      4        1.4254             nan     0.1000    0.0516
    ##      5        1.3911             nan     0.1000    0.0458
    ##      6        1.3614             nan     0.1000    0.0446
    ##      7        1.3332             nan     0.1000    0.0384
    ##      8        1.3085             nan     0.1000    0.0353
    ##      9        1.2859             nan     0.1000    0.0303
    ##     10        1.2660             nan     0.1000    0.0313
    ##     20        1.1080             nan     0.1000    0.0134
    ##     40        0.9396             nan     0.1000    0.0083
    ##     60        0.8324             nan     0.1000    0.0068
    ##     80        0.7519             nan     0.1000    0.0044
    ##    100        0.6858             nan     0.1000    0.0045
    ##    120        0.6307             nan     0.1000    0.0033
    ##    140        0.5862             nan     0.1000    0.0033
    ##    150        0.5649             nan     0.1000    0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1799
    ##      2        1.4934             nan     0.1000    0.1290
    ##      3        1.4102             nan     0.1000    0.1062
    ##      4        1.3435             nan     0.1000    0.0866
    ##      5        1.2888             nan     0.1000    0.0732
    ##      6        1.2418             nan     0.1000    0.0728
    ##      7        1.1962             nan     0.1000    0.0592
    ##      8        1.1588             nan     0.1000    0.0470
    ##      9        1.1280             nan     0.1000    0.0412
    ##     10        1.1015             nan     0.1000    0.0406
    ##     20        0.8895             nan     0.1000    0.0225
    ##     40        0.6754             nan     0.1000    0.0104
    ##     60        0.5443             nan     0.1000    0.0082
    ##     80        0.4542             nan     0.1000    0.0051
    ##    100        0.3889             nan     0.1000    0.0036
    ##    120        0.3378             nan     0.1000    0.0039
    ##    140        0.2950             nan     0.1000    0.0013
    ##    150        0.2786             nan     0.1000    0.0017
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2354
    ##      2        1.4615             nan     0.1000    0.1618
    ##      3        1.3598             nan     0.1000    0.1315
    ##      4        1.2773             nan     0.1000    0.1018
    ##      5        1.2120             nan     0.1000    0.0884
    ##      6        1.1559             nan     0.1000    0.0733
    ##      7        1.1099             nan     0.1000    0.0681
    ##      8        1.0675             nan     0.1000    0.0574
    ##      9        1.0317             nan     0.1000    0.0579
    ##     10        0.9957             nan     0.1000    0.0430
    ##     20        0.7574             nan     0.1000    0.0224
    ##     40        0.5238             nan     0.1000    0.0113
    ##     60        0.3990             nan     0.1000    0.0064
    ##     80        0.3130             nan     0.1000    0.0053
    ##    100        0.2567             nan     0.1000    0.0037
    ##    120        0.2119             nan     0.1000    0.0025
    ##    140        0.1782             nan     0.1000    0.0030
    ##    150        0.1643             nan     0.1000    0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1227
    ##      2        1.5239             nan     0.1000    0.0871
    ##      3        1.4658             nan     0.1000    0.0670
    ##      4        1.4214             nan     0.1000    0.0536
    ##      5        1.3856             nan     0.1000    0.0417
    ##      6        1.3568             nan     0.1000    0.0474
    ##      7        1.3273             nan     0.1000    0.0384
    ##      8        1.3025             nan     0.1000    0.0321
    ##      9        1.2813             nan     0.1000    0.0321
    ##     10        1.2598             nan     0.1000    0.0335
    ##     20        1.1034             nan     0.1000    0.0175
    ##     40        0.9304             nan     0.1000    0.0104
    ##     60        0.8221             nan     0.1000    0.0082
    ##     80        0.7414             nan     0.1000    0.0056
    ##    100        0.6766             nan     0.1000    0.0032
    ##    120        0.6234             nan     0.1000    0.0027
    ##    140        0.5817             nan     0.1000    0.0031
    ##    150        0.5622             nan     0.1000    0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1853
    ##      2        1.4894             nan     0.1000    0.1300
    ##      3        1.4037             nan     0.1000    0.1034
    ##      4        1.3385             nan     0.1000    0.0844
    ##      5        1.2839             nan     0.1000    0.0697
    ##      6        1.2401             nan     0.1000    0.0708
    ##      7        1.1947             nan     0.1000    0.0596
    ##      8        1.1579             nan     0.1000    0.0551
    ##      9        1.1239             nan     0.1000    0.0412
    ##     10        1.0972             nan     0.1000    0.0435
    ##     20        0.8961             nan     0.1000    0.0208
    ##     40        0.6792             nan     0.1000    0.0098
    ##     60        0.5557             nan     0.1000    0.0076
    ##     80        0.4621             nan     0.1000    0.0047
    ##    100        0.3950             nan     0.1000    0.0055
    ##    120        0.3385             nan     0.1000    0.0025
    ##    140        0.2960             nan     0.1000    0.0021
    ##    150        0.2782             nan     0.1000    0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2373
    ##      2        1.4602             nan     0.1000    0.1637
    ##      3        1.3565             nan     0.1000    0.1226
    ##      4        1.2772             nan     0.1000    0.0996
    ##      5        1.2133             nan     0.1000    0.0853
    ##      6        1.1577             nan     0.1000    0.0775
    ##      7        1.1084             nan     0.1000    0.0607
    ##      8        1.0694             nan     0.1000    0.0728
    ##      9        1.0244             nan     0.1000    0.0591
    ##     10        0.9873             nan     0.1000    0.0516
    ##     20        0.7515             nan     0.1000    0.0268
    ##     40        0.5239             nan     0.1000    0.0125
    ##     60        0.3958             nan     0.1000    0.0063
    ##     80        0.3150             nan     0.1000    0.0064
    ##    100        0.2573             nan     0.1000    0.0037
    ##    120        0.2123             nan     0.1000    0.0019
    ##    140        0.1798             nan     0.1000    0.0020
    ##    150        0.1656             nan     0.1000    0.0009
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1248
    ##      2        1.5237             nan     0.1000    0.0882
    ##      3        1.4660             nan     0.1000    0.0660
    ##      4        1.4224             nan     0.1000    0.0515
    ##      5        1.3880             nan     0.1000    0.0508
    ##      6        1.3546             nan     0.1000    0.0456
    ##      7        1.3256             nan     0.1000    0.0370
    ##      8        1.3022             nan     0.1000    0.0310
    ##      9        1.2810             nan     0.1000    0.0340
    ##     10        1.2589             nan     0.1000    0.0310
    ##     20        1.1041             nan     0.1000    0.0167
    ##     40        0.9321             nan     0.1000    0.0092
    ##     60        0.8257             nan     0.1000    0.0044
    ##     80        0.7442             nan     0.1000    0.0055
    ##    100        0.6794             nan     0.1000    0.0032
    ##    120        0.6257             nan     0.1000    0.0034
    ##    140        0.5787             nan     0.1000    0.0025
    ##    150        0.5583             nan     0.1000    0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1911
    ##      2        1.4875             nan     0.1000    0.1281
    ##      3        1.4053             nan     0.1000    0.1061
    ##      4        1.3377             nan     0.1000    0.0878
    ##      5        1.2815             nan     0.1000    0.0774
    ##      6        1.2325             nan     0.1000    0.0593
    ##      7        1.1955             nan     0.1000    0.0595
    ##      8        1.1578             nan     0.1000    0.0467
    ##      9        1.1281             nan     0.1000    0.0508
    ##     10        1.0953             nan     0.1000    0.0436
    ##     20        0.8910             nan     0.1000    0.0198
    ##     40        0.6793             nan     0.1000    0.0084
    ##     60        0.5480             nan     0.1000    0.0084
    ##     80        0.4542             nan     0.1000    0.0040
    ##    100        0.3902             nan     0.1000    0.0042
    ##    120        0.3363             nan     0.1000    0.0022
    ##    140        0.2952             nan     0.1000    0.0025
    ##    150        0.2776             nan     0.1000    0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2396
    ##      2        1.4573             nan     0.1000    0.1659
    ##      3        1.3550             nan     0.1000    0.1259
    ##      4        1.2744             nan     0.1000    0.1118
    ##      5        1.2058             nan     0.1000    0.0854
    ##      6        1.1531             nan     0.1000    0.0714
    ##      7        1.1065             nan     0.1000    0.0738
    ##      8        1.0608             nan     0.1000    0.0610
    ##      9        1.0218             nan     0.1000    0.0469
    ##     10        0.9914             nan     0.1000    0.0546
    ##     20        0.7509             nan     0.1000    0.0262
    ##     40        0.5233             nan     0.1000    0.0106
    ##     60        0.3939             nan     0.1000    0.0084
    ##     80        0.3116             nan     0.1000    0.0038
    ##    100        0.2533             nan     0.1000    0.0034
    ##    120        0.2122             nan     0.1000    0.0016
    ##    140        0.1796             nan     0.1000    0.0016
    ##    150        0.1656             nan     0.1000    0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1242
    ##      2        1.5248             nan     0.1000    0.0828
    ##      3        1.4688             nan     0.1000    0.0647
    ##      4        1.4255             nan     0.1000    0.0520
    ##      5        1.3915             nan     0.1000    0.0414
    ##      6        1.3639             nan     0.1000    0.0438
    ##      7        1.3362             nan     0.1000    0.0377
    ##      8        1.3116             nan     0.1000    0.0372
    ##      9        1.2876             nan     0.1000    0.0354
    ##     10        1.2663             nan     0.1000    0.0285
    ##     20        1.1129             nan     0.1000    0.0181
    ##     40        0.9398             nan     0.1000    0.0083
    ##     60        0.8321             nan     0.1000    0.0063
    ##     80        0.7514             nan     0.1000    0.0050
    ##    100        0.6901             nan     0.1000    0.0053
    ##    120        0.6350             nan     0.1000    0.0035
    ##    140        0.5917             nan     0.1000    0.0032
    ##    150        0.5715             nan     0.1000    0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1774
    ##      2        1.4947             nan     0.1000    0.1343
    ##      3        1.4093             nan     0.1000    0.1026
    ##      4        1.3448             nan     0.1000    0.0837
    ##      5        1.2911             nan     0.1000    0.0783
    ##      6        1.2423             nan     0.1000    0.0534
    ##      7        1.2077             nan     0.1000    0.0634
    ##      8        1.1681             nan     0.1000    0.0518
    ##      9        1.1349             nan     0.1000    0.0434
    ##     10        1.1070             nan     0.1000    0.0370
    ##     20        0.9068             nan     0.1000    0.0273
    ##     40        0.6948             nan     0.1000    0.0193
    ##     60        0.5586             nan     0.1000    0.0056
    ##     80        0.4689             nan     0.1000    0.0049
    ##    100        0.4010             nan     0.1000    0.0049
    ##    120        0.3481             nan     0.1000    0.0029
    ##    140        0.3041             nan     0.1000    0.0021
    ##    150        0.2860             nan     0.1000    0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2320
    ##      2        1.4615             nan     0.1000    0.1589
    ##      3        1.3619             nan     0.1000    0.1278
    ##      4        1.2823             nan     0.1000    0.1042
    ##      5        1.2158             nan     0.1000    0.0833
    ##      6        1.1626             nan     0.1000    0.0692
    ##      7        1.1192             nan     0.1000    0.0718
    ##      8        1.0727             nan     0.1000    0.0660
    ##      9        1.0313             nan     0.1000    0.0630
    ##     10        0.9928             nan     0.1000    0.0442
    ##     20        0.7675             nan     0.1000    0.0222
    ##     40        0.5393             nan     0.1000    0.0097
    ##     60        0.4064             nan     0.1000    0.0092
    ##     80        0.3228             nan     0.1000    0.0046
    ##    100        0.2645             nan     0.1000    0.0020
    ##    120        0.2247             nan     0.1000    0.0033
    ##    140        0.1888             nan     0.1000    0.0030
    ##    150        0.1742             nan     0.1000    0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1299
    ##      2        1.5230             nan     0.1000    0.0900
    ##      3        1.4622             nan     0.1000    0.0674
    ##      4        1.4175             nan     0.1000    0.0522
    ##      5        1.3816             nan     0.1000    0.0437
    ##      6        1.3531             nan     0.1000    0.0413
    ##      7        1.3253             nan     0.1000    0.0407
    ##      8        1.2992             nan     0.1000    0.0358
    ##      9        1.2764             nan     0.1000    0.0367
    ##     10        1.2521             nan     0.1000    0.0297
    ##     20        1.0986             nan     0.1000    0.0185
    ##     40        0.9220             nan     0.1000    0.0084
    ##     60        0.8125             nan     0.1000    0.0067
    ##     80        0.7314             nan     0.1000    0.0060
    ##    100        0.6665             nan     0.1000    0.0037
    ##    120        0.6150             nan     0.1000    0.0033
    ##    140        0.5707             nan     0.1000    0.0025
    ##    150        0.5493             nan     0.1000    0.0028
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1851
    ##      2        1.4879             nan     0.1000    0.1402
    ##      3        1.4005             nan     0.1000    0.1009
    ##      4        1.3353             nan     0.1000    0.0815
    ##      5        1.2822             nan     0.1000    0.0690
    ##      6        1.2376             nan     0.1000    0.0691
    ##      7        1.1935             nan     0.1000    0.0621
    ##      8        1.1538             nan     0.1000    0.0486
    ##      9        1.1226             nan     0.1000    0.0492
    ##     10        1.0911             nan     0.1000    0.0446
    ##     20        0.8899             nan     0.1000    0.0205
    ##     40        0.6690             nan     0.1000    0.0145
    ##     60        0.5393             nan     0.1000    0.0079
    ##     80        0.4524             nan     0.1000    0.0050
    ##    100        0.3840             nan     0.1000    0.0052
    ##    120        0.3296             nan     0.1000    0.0034
    ##    140        0.2896             nan     0.1000    0.0033
    ##    150        0.2721             nan     0.1000    0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2392
    ##      2        1.4571             nan     0.1000    0.1669
    ##      3        1.3531             nan     0.1000    0.1339
    ##      4        1.2692             nan     0.1000    0.1012
    ##      5        1.2053             nan     0.1000    0.0925
    ##      6        1.1490             nan     0.1000    0.0685
    ##      7        1.1053             nan     0.1000    0.0758
    ##      8        1.0579             nan     0.1000    0.0588
    ##      9        1.0210             nan     0.1000    0.0529
    ##     10        0.9863             nan     0.1000    0.0616
    ##     20        0.7448             nan     0.1000    0.0265
    ##     40        0.5186             nan     0.1000    0.0125
    ##     60        0.3951             nan     0.1000    0.0092
    ##     80        0.3077             nan     0.1000    0.0061
    ##    100        0.2470             nan     0.1000    0.0025
    ##    120        0.2073             nan     0.1000    0.0018
    ##    140        0.1766             nan     0.1000    0.0012
    ##    150        0.1639             nan     0.1000    0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1257
    ##      2        1.5232             nan     0.1000    0.0899
    ##      3        1.4652             nan     0.1000    0.0657
    ##      4        1.4220             nan     0.1000    0.0539
    ##      5        1.3865             nan     0.1000    0.0508
    ##      6        1.3533             nan     0.1000    0.0419
    ##      7        1.3266             nan     0.1000    0.0350
    ##      8        1.3041             nan     0.1000    0.0372
    ##      9        1.2810             nan     0.1000    0.0310
    ##     10        1.2597             nan     0.1000    0.0296
    ##     20        1.1078             nan     0.1000    0.0174
    ##     40        0.9386             nan     0.1000    0.0091
    ##     60        0.8288             nan     0.1000    0.0066
    ##     80        0.7467             nan     0.1000    0.0041
    ##    100        0.6828             nan     0.1000    0.0037
    ##    120        0.6306             nan     0.1000    0.0028
    ##    140        0.5853             nan     0.1000    0.0021
    ##    150        0.5656             nan     0.1000    0.0032
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1902
    ##      2        1.4861             nan     0.1000    0.1263
    ##      3        1.4038             nan     0.1000    0.1067
    ##      4        1.3359             nan     0.1000    0.0836
    ##      5        1.2836             nan     0.1000    0.0664
    ##      6        1.2410             nan     0.1000    0.0687
    ##      7        1.1977             nan     0.1000    0.0597
    ##      8        1.1591             nan     0.1000    0.0556
    ##      9        1.1244             nan     0.1000    0.0418
    ##     10        1.0969             nan     0.1000    0.0449
    ##     20        0.8978             nan     0.1000    0.0240
    ##     40        0.6797             nan     0.1000    0.0124
    ##     60        0.5520             nan     0.1000    0.0055
    ##     80        0.4673             nan     0.1000    0.0046
    ##    100        0.3965             nan     0.1000    0.0047
    ##    120        0.3446             nan     0.1000    0.0032
    ##    140        0.3044             nan     0.1000    0.0035
    ##    150        0.2855             nan     0.1000    0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2371
    ##      2        1.4602             nan     0.1000    0.1610
    ##      3        1.3577             nan     0.1000    0.1186
    ##      4        1.2816             nan     0.1000    0.1008
    ##      5        1.2171             nan     0.1000    0.0887
    ##      6        1.1625             nan     0.1000    0.0755
    ##      7        1.1143             nan     0.1000    0.0703
    ##      8        1.0690             nan     0.1000    0.0602
    ##      9        1.0314             nan     0.1000    0.0666
    ##     10        0.9917             nan     0.1000    0.0537
    ##     20        0.7631             nan     0.1000    0.0263
    ##     40        0.5294             nan     0.1000    0.0124
    ##     60        0.3976             nan     0.1000    0.0077
    ##     80        0.3171             nan     0.1000    0.0049
    ##    100        0.2574             nan     0.1000    0.0023
    ##    120        0.2137             nan     0.1000    0.0023
    ##    140        0.1787             nan     0.1000    0.0016
    ##    150        0.1654             nan     0.1000    0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1398
    ##      2        1.5214             nan     0.1000    0.0911
    ##      3        1.4612             nan     0.1000    0.0700
    ##      4        1.4148             nan     0.1000    0.0550
    ##      5        1.3787             nan     0.1000    0.0471
    ##      6        1.3468             nan     0.1000    0.0406
    ##      7        1.3198             nan     0.1000    0.0358
    ##      8        1.2962             nan     0.1000    0.0402
    ##      9        1.2710             nan     0.1000    0.0332
    ##     10        1.2496             nan     0.1000    0.0312
    ##     20        1.0931             nan     0.1000    0.0195
    ##     40        0.9212             nan     0.1000    0.0082
    ##     60        0.8137             nan     0.1000    0.0065
    ##     80        0.7317             nan     0.1000    0.0033
    ##    100        0.6713             nan     0.1000    0.0045
    ##    120        0.6209             nan     0.1000    0.0031
    ##    140        0.5759             nan     0.1000    0.0031
    ##    150        0.5560             nan     0.1000    0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1912
    ##      2        1.4854             nan     0.1000    0.1314
    ##      3        1.4012             nan     0.1000    0.1085
    ##      4        1.3325             nan     0.1000    0.0835
    ##      5        1.2798             nan     0.1000    0.0706
    ##      6        1.2349             nan     0.1000    0.0758
    ##      7        1.1880             nan     0.1000    0.0581
    ##      8        1.1513             nan     0.1000    0.0560
    ##      9        1.1157             nan     0.1000    0.0484
    ##     10        1.0852             nan     0.1000    0.0422
    ##     20        0.8796             nan     0.1000    0.0243
    ##     40        0.6677             nan     0.1000    0.0083
    ##     60        0.5442             nan     0.1000    0.0103
    ##     80        0.4574             nan     0.1000    0.0063
    ##    100        0.3908             nan     0.1000    0.0049
    ##    120        0.3415             nan     0.1000    0.0031
    ##    140        0.2985             nan     0.1000    0.0024
    ##    150        0.2808             nan     0.1000    0.0014
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2419
    ##      2        1.4572             nan     0.1000    0.1753
    ##      3        1.3487             nan     0.1000    0.1222
    ##      4        1.2716             nan     0.1000    0.1077
    ##      5        1.2032             nan     0.1000    0.0984
    ##      6        1.1408             nan     0.1000    0.0783
    ##      7        1.0907             nan     0.1000    0.0636
    ##      8        1.0497             nan     0.1000    0.0567
    ##      9        1.0138             nan     0.1000    0.0597
    ##     10        0.9768             nan     0.1000    0.0509
    ##     20        0.7390             nan     0.1000    0.0262
    ##     40        0.5120             nan     0.1000    0.0088
    ##     60        0.3940             nan     0.1000    0.0057
    ##     80        0.3122             nan     0.1000    0.0037
    ##    100        0.2521             nan     0.1000    0.0017
    ##    120        0.2099             nan     0.1000    0.0015
    ##    140        0.1763             nan     0.1000    0.0014
    ##    150        0.1633             nan     0.1000    0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1312
    ##      2        1.5215             nan     0.1000    0.0875
    ##      3        1.4641             nan     0.1000    0.0667
    ##      4        1.4193             nan     0.1000    0.0521
    ##      5        1.3842             nan     0.1000    0.0482
    ##      6        1.3519             nan     0.1000    0.0431
    ##      7        1.3251             nan     0.1000    0.0411
    ##      8        1.2995             nan     0.1000    0.0316
    ##      9        1.2783             nan     0.1000    0.0279
    ##     10        1.2591             nan     0.1000    0.0338
    ##     20        1.1018             nan     0.1000    0.0134
    ##     40        0.9316             nan     0.1000    0.0082
    ##     60        0.8238             nan     0.1000    0.0069
    ##     80        0.7436             nan     0.1000    0.0041
    ##    100        0.6793             nan     0.1000    0.0033
    ##    120        0.6291             nan     0.1000    0.0028
    ##    140        0.5841             nan     0.1000    0.0020
    ##    150        0.5647             nan     0.1000    0.0029
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1911
    ##      2        1.4848             nan     0.1000    0.1331
    ##      3        1.3992             nan     0.1000    0.1012
    ##      4        1.3337             nan     0.1000    0.0842
    ##      5        1.2798             nan     0.1000    0.0673
    ##      6        1.2357             nan     0.1000    0.0593
    ##      7        1.1987             nan     0.1000    0.0618
    ##      8        1.1591             nan     0.1000    0.0527
    ##      9        1.1253             nan     0.1000    0.0427
    ##     10        1.0975             nan     0.1000    0.0427
    ##     20        0.8914             nan     0.1000    0.0206
    ##     40        0.6788             nan     0.1000    0.0122
    ##     60        0.5493             nan     0.1000    0.0091
    ##     80        0.4627             nan     0.1000    0.0066
    ##    100        0.3928             nan     0.1000    0.0042
    ##    120        0.3392             nan     0.1000    0.0038
    ##    140        0.2972             nan     0.1000    0.0030
    ##    150        0.2791             nan     0.1000    0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2380
    ##      2        1.4568             nan     0.1000    0.1560
    ##      3        1.3535             nan     0.1000    0.1312
    ##      4        1.2721             nan     0.1000    0.1058
    ##      5        1.2059             nan     0.1000    0.0820
    ##      6        1.1530             nan     0.1000    0.0708
    ##      7        1.1087             nan     0.1000    0.0750
    ##      8        1.0622             nan     0.1000    0.0589
    ##      9        1.0252             nan     0.1000    0.0472
    ##     10        0.9938             nan     0.1000    0.0554
    ##     20        0.7564             nan     0.1000    0.0242
    ##     40        0.5221             nan     0.1000    0.0098
    ##     60        0.3992             nan     0.1000    0.0067
    ##     80        0.3181             nan     0.1000    0.0069
    ##    100        0.2597             nan     0.1000    0.0030
    ##    120        0.2161             nan     0.1000    0.0021
    ##    140        0.1824             nan     0.1000    0.0011
    ##    150        0.1683             nan     0.1000    0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1234
    ##      2        1.5241             nan     0.1000    0.0885
    ##      3        1.4657             nan     0.1000    0.0670
    ##      4        1.4213             nan     0.1000    0.0543
    ##      5        1.3857             nan     0.1000    0.0489
    ##      6        1.3538             nan     0.1000    0.0436
    ##      7        1.3248             nan     0.1000    0.0405
    ##      8        1.2988             nan     0.1000    0.0311
    ##      9        1.2777             nan     0.1000    0.0324
    ##     10        1.2575             nan     0.1000    0.0300
    ##     20        1.1046             nan     0.1000    0.0191
    ##     40        0.9284             nan     0.1000    0.0100
    ##     60        0.8200             nan     0.1000    0.0054
    ##     80        0.7412             nan     0.1000    0.0051
    ##    100        0.6770             nan     0.1000    0.0037
    ##    120        0.6244             nan     0.1000    0.0031
    ##    140        0.5791             nan     0.1000    0.0020
    ##    150        0.5602             nan     0.1000    0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1916
    ##      2        1.4864             nan     0.1000    0.1304
    ##      3        1.4007             nan     0.1000    0.0975
    ##      4        1.3365             nan     0.1000    0.0839
    ##      5        1.2830             nan     0.1000    0.0775
    ##      6        1.2320             nan     0.1000    0.0712
    ##      7        1.1880             nan     0.1000    0.0507
    ##      8        1.1548             nan     0.1000    0.0573
    ##      9        1.1191             nan     0.1000    0.0453
    ##     10        1.0902             nan     0.1000    0.0477
    ##     20        0.8887             nan     0.1000    0.0202
    ##     40        0.6774             nan     0.1000    0.0163
    ##     60        0.5437             nan     0.1000    0.0050
    ##     80        0.4544             nan     0.1000    0.0052
    ##    100        0.3867             nan     0.1000    0.0049
    ##    120        0.3329             nan     0.1000    0.0019
    ##    140        0.2919             nan     0.1000    0.0023
    ##    150        0.2720             nan     0.1000    0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2378
    ##      2        1.4560             nan     0.1000    0.1633
    ##      3        1.3521             nan     0.1000    0.1232
    ##      4        1.2727             nan     0.1000    0.1069
    ##      5        1.2051             nan     0.1000    0.0845
    ##      6        1.1510             nan     0.1000    0.0792
    ##      7        1.1007             nan     0.1000    0.0666
    ##      8        1.0580             nan     0.1000    0.0709
    ##      9        1.0146             nan     0.1000    0.0546
    ##     10        0.9800             nan     0.1000    0.0472
    ##     20        0.7502             nan     0.1000    0.0271
    ##     40        0.5213             nan     0.1000    0.0098
    ##     60        0.3931             nan     0.1000    0.0052
    ##     80        0.3080             nan     0.1000    0.0054
    ##    100        0.2536             nan     0.1000    0.0034
    ##    120        0.2093             nan     0.1000    0.0029
    ##    140        0.1757             nan     0.1000    0.0014
    ##    150        0.1619             nan     0.1000    0.0017
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1293
    ##      2        1.5233             nan     0.1000    0.0865
    ##      3        1.4655             nan     0.1000    0.0652
    ##      4        1.4220             nan     0.1000    0.0524
    ##      5        1.3870             nan     0.1000    0.0486
    ##      6        1.3546             nan     0.1000    0.0368
    ##      7        1.3297             nan     0.1000    0.0430
    ##      8        1.3029             nan     0.1000    0.0308
    ##      9        1.2824             nan     0.1000    0.0308
    ##     10        1.2626             nan     0.1000    0.0273
    ##     20        1.1065             nan     0.1000    0.0202
    ##     40        0.9320             nan     0.1000    0.0105
    ##     60        0.8218             nan     0.1000    0.0085
    ##     80        0.7402             nan     0.1000    0.0054
    ##    100        0.6781             nan     0.1000    0.0029
    ##    120        0.6252             nan     0.1000    0.0030
    ##    140        0.5816             nan     0.1000    0.0024
    ##    150        0.5617             nan     0.1000    0.0027
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1897
    ##      2        1.4853             nan     0.1000    0.1287
    ##      3        1.4028             nan     0.1000    0.1090
    ##      4        1.3345             nan     0.1000    0.0847
    ##      5        1.2809             nan     0.1000    0.0701
    ##      6        1.2361             nan     0.1000    0.0627
    ##      7        1.1956             nan     0.1000    0.0646
    ##      8        1.1559             nan     0.1000    0.0481
    ##      9        1.1250             nan     0.1000    0.0456
    ##     10        1.0960             nan     0.1000    0.0362
    ##     20        0.8948             nan     0.1000    0.0197
    ##     40        0.6859             nan     0.1000    0.0126
    ##     60        0.5507             nan     0.1000    0.0102
    ##     80        0.4603             nan     0.1000    0.0057
    ##    100        0.3896             nan     0.1000    0.0040
    ##    120        0.3401             nan     0.1000    0.0026
    ##    140        0.2982             nan     0.1000    0.0018
    ##    150        0.2797             nan     0.1000    0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2336
    ##      2        1.4622             nan     0.1000    0.1611
    ##      3        1.3599             nan     0.1000    0.1263
    ##      4        1.2783             nan     0.1000    0.1007
    ##      5        1.2140             nan     0.1000    0.0924
    ##      6        1.1557             nan     0.1000    0.0702
    ##      7        1.1105             nan     0.1000    0.0674
    ##      8        1.0683             nan     0.1000    0.0646
    ##      9        1.0270             nan     0.1000    0.0501
    ##     10        0.9947             nan     0.1000    0.0496
    ##     20        0.7544             nan     0.1000    0.0209
    ##     40        0.5204             nan     0.1000    0.0123
    ##     60        0.3976             nan     0.1000    0.0127
    ##     80        0.3114             nan     0.1000    0.0053
    ##    100        0.2532             nan     0.1000    0.0029
    ##    120        0.2101             nan     0.1000    0.0028
    ##    140        0.1790             nan     0.1000    0.0021
    ##    150        0.1646             nan     0.1000    0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1335
    ##      2        1.5218             nan     0.1000    0.0866
    ##      3        1.4636             nan     0.1000    0.0688
    ##      4        1.4193             nan     0.1000    0.0501
    ##      5        1.3851             nan     0.1000    0.0541
    ##      6        1.3517             nan     0.1000    0.0421
    ##      7        1.3240             nan     0.1000    0.0370
    ##      8        1.3007             nan     0.1000    0.0331
    ##      9        1.2788             nan     0.1000    0.0347
    ##     10        1.2567             nan     0.1000    0.0323
    ##     20        1.1013             nan     0.1000    0.0157
    ##     40        0.9311             nan     0.1000    0.0094
    ##     60        0.8228             nan     0.1000    0.0061
    ##     80        0.7418             nan     0.1000    0.0047
    ##    100        0.6796             nan     0.1000    0.0030
    ##    120        0.6279             nan     0.1000    0.0041
    ##    140        0.5830             nan     0.1000    0.0020
    ##    150        0.5644             nan     0.1000    0.0030
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1826
    ##      2        1.4911             nan     0.1000    0.1387
    ##      3        1.4030             nan     0.1000    0.0946
    ##      4        1.3410             nan     0.1000    0.0897
    ##      5        1.2850             nan     0.1000    0.0796
    ##      6        1.2356             nan     0.1000    0.0654
    ##      7        1.1943             nan     0.1000    0.0655
    ##      8        1.1530             nan     0.1000    0.0458
    ##      9        1.1239             nan     0.1000    0.0486
    ##     10        1.0933             nan     0.1000    0.0421
    ##     20        0.8862             nan     0.1000    0.0199
    ##     40        0.6813             nan     0.1000    0.0181
    ##     60        0.5486             nan     0.1000    0.0051
    ##     80        0.4620             nan     0.1000    0.0034
    ##    100        0.3957             nan     0.1000    0.0039
    ##    120        0.3409             nan     0.1000    0.0041
    ##    140        0.2975             nan     0.1000    0.0020
    ##    150        0.2791             nan     0.1000    0.0012
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2364
    ##      2        1.4581             nan     0.1000    0.1665
    ##      3        1.3539             nan     0.1000    0.1235
    ##      4        1.2757             nan     0.1000    0.1118
    ##      5        1.2060             nan     0.1000    0.0841
    ##      6        1.1523             nan     0.1000    0.0843
    ##      7        1.1004             nan     0.1000    0.0591
    ##      8        1.0617             nan     0.1000    0.0651
    ##      9        1.0209             nan     0.1000    0.0636
    ##     10        0.9824             nan     0.1000    0.0477
    ##     20        0.7534             nan     0.1000    0.0280
    ##     40        0.5242             nan     0.1000    0.0138
    ##     60        0.3950             nan     0.1000    0.0075
    ##     80        0.3140             nan     0.1000    0.0056
    ##    100        0.2567             nan     0.1000    0.0023
    ##    120        0.2138             nan     0.1000    0.0019
    ##    140        0.1807             nan     0.1000    0.0021
    ##    150        0.1657             nan     0.1000    0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1282
    ##      2        1.5235             nan     0.1000    0.0848
    ##      3        1.4669             nan     0.1000    0.0618
    ##      4        1.4249             nan     0.1000    0.0552
    ##      5        1.3894             nan     0.1000    0.0444
    ##      6        1.3604             nan     0.1000    0.0411
    ##      7        1.3334             nan     0.1000    0.0360
    ##      8        1.3105             nan     0.1000    0.0371
    ##      9        1.2876             nan     0.1000    0.0296
    ##     10        1.2681             nan     0.1000    0.0331
    ##     20        1.1112             nan     0.1000    0.0191
    ##     40        0.9379             nan     0.1000    0.0093
    ##     60        0.8288             nan     0.1000    0.0065
    ##     80        0.7475             nan     0.1000    0.0051
    ##    100        0.6833             nan     0.1000    0.0039
    ##    120        0.6304             nan     0.1000    0.0028
    ##    140        0.5829             nan     0.1000    0.0028
    ##    150        0.5625             nan     0.1000    0.0025
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1789
    ##      2        1.4916             nan     0.1000    0.1341
    ##      3        1.4050             nan     0.1000    0.1011
    ##      4        1.3387             nan     0.1000    0.0875
    ##      5        1.2831             nan     0.1000    0.0672
    ##      6        1.2390             nan     0.1000    0.0560
    ##      7        1.2026             nan     0.1000    0.0631
    ##      8        1.1632             nan     0.1000    0.0566
    ##      9        1.1270             nan     0.1000    0.0411
    ##     10        1.1003             nan     0.1000    0.0378
    ##     20        0.8952             nan     0.1000    0.0212
    ##     40        0.6737             nan     0.1000    0.0128
    ##     60        0.5465             nan     0.1000    0.0066
    ##     80        0.4601             nan     0.1000    0.0069
    ##    100        0.3903             nan     0.1000    0.0042
    ##    120        0.3381             nan     0.1000    0.0038
    ##    140        0.2926             nan     0.1000    0.0022
    ##    150        0.2744             nan     0.1000    0.0018
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2408
    ##      2        1.4584             nan     0.1000    0.1620
    ##      3        1.3575             nan     0.1000    0.1219
    ##      4        1.2800             nan     0.1000    0.0949
    ##      5        1.2177             nan     0.1000    0.0930
    ##      6        1.1599             nan     0.1000    0.0761
    ##      7        1.1112             nan     0.1000    0.0675
    ##      8        1.0676             nan     0.1000    0.0530
    ##      9        1.0325             nan     0.1000    0.0499
    ##     10        0.9998             nan     0.1000    0.0611
    ##     20        0.7608             nan     0.1000    0.0244
    ##     40        0.5226             nan     0.1000    0.0137
    ##     60        0.3973             nan     0.1000    0.0079
    ##     80        0.3176             nan     0.1000    0.0065
    ##    100        0.2566             nan     0.1000    0.0031
    ##    120        0.2116             nan     0.1000    0.0026
    ##    140        0.1777             nan     0.1000    0.0018
    ##    150        0.1633             nan     0.1000    0.0013
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1343
    ##      2        1.5207             nan     0.1000    0.0869
    ##      3        1.4619             nan     0.1000    0.0642
    ##      4        1.4174             nan     0.1000    0.0578
    ##      5        1.3804             nan     0.1000    0.0530
    ##      6        1.3466             nan     0.1000    0.0431
    ##      7        1.3186             nan     0.1000    0.0347
    ##      8        1.2961             nan     0.1000    0.0356
    ##      9        1.2727             nan     0.1000    0.0330
    ##     10        1.2505             nan     0.1000    0.0322
    ##     20        1.0961             nan     0.1000    0.0177
    ##     40        0.9242             nan     0.1000    0.0098
    ##     60        0.8183             nan     0.1000    0.0061
    ##     80        0.7363             nan     0.1000    0.0056
    ##    100        0.6716             nan     0.1000    0.0032
    ##    120        0.6196             nan     0.1000    0.0027
    ##    140        0.5765             nan     0.1000    0.0022
    ##    150        0.5556             nan     0.1000    0.0011
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1920
    ##      2        1.4857             nan     0.1000    0.1324
    ##      3        1.4013             nan     0.1000    0.1067
    ##      4        1.3346             nan     0.1000    0.0829
    ##      5        1.2819             nan     0.1000    0.0701
    ##      6        1.2371             nan     0.1000    0.0736
    ##      7        1.1919             nan     0.1000    0.0655
    ##      8        1.1510             nan     0.1000    0.0515
    ##      9        1.1188             nan     0.1000    0.0410
    ##     10        1.0923             nan     0.1000    0.0420
    ##     20        0.8892             nan     0.1000    0.0205
    ##     40        0.6798             nan     0.1000    0.0092
    ##     60        0.5516             nan     0.1000    0.0104
    ##     80        0.4556             nan     0.1000    0.0037
    ##    100        0.3911             nan     0.1000    0.0033
    ##    120        0.3362             nan     0.1000    0.0027
    ##    140        0.2933             nan     0.1000    0.0023
    ##    150        0.2753             nan     0.1000    0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2412
    ##      2        1.4558             nan     0.1000    0.1704
    ##      3        1.3503             nan     0.1000    0.1244
    ##      4        1.2724             nan     0.1000    0.1137
    ##      5        1.2021             nan     0.1000    0.0832
    ##      6        1.1488             nan     0.1000    0.0721
    ##      7        1.1031             nan     0.1000    0.0651
    ##      8        1.0607             nan     0.1000    0.0678
    ##      9        1.0187             nan     0.1000    0.0546
    ##     10        0.9837             nan     0.1000    0.0561
    ##     20        0.7488             nan     0.1000    0.0271
    ##     40        0.5180             nan     0.1000    0.0075
    ##     60        0.3937             nan     0.1000    0.0068
    ##     80        0.3126             nan     0.1000    0.0033
    ##    100        0.2537             nan     0.1000    0.0037
    ##    120        0.2084             nan     0.1000    0.0022
    ##    140        0.1769             nan     0.1000    0.0018
    ##    150        0.1625             nan     0.1000    0.0016
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1300
    ##      2        1.5240             nan     0.1000    0.0861
    ##      3        1.4673             nan     0.1000    0.0648
    ##      4        1.4242             nan     0.1000    0.0524
    ##      5        1.3901             nan     0.1000    0.0501
    ##      6        1.3572             nan     0.1000    0.0387
    ##      7        1.3322             nan     0.1000    0.0366
    ##      8        1.3081             nan     0.1000    0.0367
    ##      9        1.2851             nan     0.1000    0.0323
    ##     10        1.2638             nan     0.1000    0.0297
    ##     20        1.1079             nan     0.1000    0.0199
    ##     40        0.9343             nan     0.1000    0.0096
    ##     60        0.8243             nan     0.1000    0.0061
    ##     80        0.7449             nan     0.1000    0.0040
    ##    100        0.6826             nan     0.1000    0.0047
    ##    120        0.6255             nan     0.1000    0.0036
    ##    140        0.5790             nan     0.1000    0.0027
    ##    150        0.5611             nan     0.1000    0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1831
    ##      2        1.4878             nan     0.1000    0.1249
    ##      3        1.4064             nan     0.1000    0.1053
    ##      4        1.3401             nan     0.1000    0.0839
    ##      5        1.2857             nan     0.1000    0.0698
    ##      6        1.2412             nan     0.1000    0.0737
    ##      7        1.1958             nan     0.1000    0.0455
    ##      8        1.1655             nan     0.1000    0.0603
    ##      9        1.1277             nan     0.1000    0.0446
    ##     10        1.0998             nan     0.1000    0.0491
    ##     20        0.8931             nan     0.1000    0.0207
    ##     40        0.6745             nan     0.1000    0.0116
    ##     60        0.5449             nan     0.1000    0.0057
    ##     80        0.4570             nan     0.1000    0.0061
    ##    100        0.3883             nan     0.1000    0.0044
    ##    120        0.3368             nan     0.1000    0.0018
    ##    140        0.2963             nan     0.1000    0.0018
    ##    150        0.2788             nan     0.1000    0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2351
    ##      2        1.4581             nan     0.1000    0.1634
    ##      3        1.3539             nan     0.1000    0.1235
    ##      4        1.2758             nan     0.1000    0.1058
    ##      5        1.2088             nan     0.1000    0.0827
    ##      6        1.1564             nan     0.1000    0.0758
    ##      7        1.1088             nan     0.1000    0.0619
    ##      8        1.0688             nan     0.1000    0.0681
    ##      9        1.0263             nan     0.1000    0.0681
    ##     10        0.9858             nan     0.1000    0.0521
    ##     20        0.7503             nan     0.1000    0.0292
    ##     40        0.5202             nan     0.1000    0.0162
    ##     60        0.3946             nan     0.1000    0.0081
    ##     80        0.3111             nan     0.1000    0.0040
    ##    100        0.2547             nan     0.1000    0.0036
    ##    120        0.2078             nan     0.1000    0.0027
    ##    140        0.1747             nan     0.1000    0.0013
    ##    150        0.1630             nan     0.1000    0.0019
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1279
    ##      2        1.5251             nan     0.1000    0.0859
    ##      3        1.4678             nan     0.1000    0.0681
    ##      4        1.4243             nan     0.1000    0.0536
    ##      5        1.3890             nan     0.1000    0.0483
    ##      6        1.3580             nan     0.1000    0.0385
    ##      7        1.3326             nan     0.1000    0.0375
    ##      8        1.3082             nan     0.1000    0.0356
    ##      9        1.2852             nan     0.1000    0.0334
    ##     10        1.2628             nan     0.1000    0.0286
    ##     20        1.1068             nan     0.1000    0.0182
    ##     40        0.9346             nan     0.1000    0.0090
    ##     60        0.8272             nan     0.1000    0.0065
    ##     80        0.7472             nan     0.1000    0.0048
    ##    100        0.6808             nan     0.1000    0.0026
    ##    120        0.6314             nan     0.1000    0.0057
    ##    140        0.5844             nan     0.1000    0.0014
    ##    150        0.5644             nan     0.1000    0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1889
    ##      2        1.4890             nan     0.1000    0.1318
    ##      3        1.4044             nan     0.1000    0.0937
    ##      4        1.3424             nan     0.1000    0.0859
    ##      5        1.2877             nan     0.1000    0.0714
    ##      6        1.2421             nan     0.1000    0.0672
    ##      7        1.1992             nan     0.1000    0.0619
    ##      8        1.1592             nan     0.1000    0.0486
    ##      9        1.1282             nan     0.1000    0.0527
    ##     10        1.0951             nan     0.1000    0.0497
    ##     20        0.8908             nan     0.1000    0.0193
    ##     40        0.6756             nan     0.1000    0.0099
    ##     60        0.5448             nan     0.1000    0.0077
    ##     80        0.4568             nan     0.1000    0.0054
    ##    100        0.3901             nan     0.1000    0.0031
    ##    120        0.3377             nan     0.1000    0.0033
    ##    140        0.2941             nan     0.1000    0.0024
    ##    150        0.2763             nan     0.1000    0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2327
    ##      2        1.4624             nan     0.1000    0.1668
    ##      3        1.3592             nan     0.1000    0.1286
    ##      4        1.2791             nan     0.1000    0.0993
    ##      5        1.2158             nan     0.1000    0.0953
    ##      6        1.1558             nan     0.1000    0.0646
    ##      7        1.1134             nan     0.1000    0.0662
    ##      8        1.0714             nan     0.1000    0.0654
    ##      9        1.0305             nan     0.1000    0.0571
    ##     10        0.9941             nan     0.1000    0.0527
    ##     20        0.7583             nan     0.1000    0.0275
    ##     40        0.5248             nan     0.1000    0.0112
    ##     60        0.3963             nan     0.1000    0.0085
    ##     80        0.3158             nan     0.1000    0.0042
    ##    100        0.2587             nan     0.1000    0.0029
    ##    120        0.2157             nan     0.1000    0.0025
    ##    140        0.1833             nan     0.1000    0.0017
    ##    150        0.1694             nan     0.1000    0.0021
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1298
    ##      2        1.5218             nan     0.1000    0.0878
    ##      3        1.4629             nan     0.1000    0.0665
    ##      4        1.4186             nan     0.1000    0.0563
    ##      5        1.3827             nan     0.1000    0.0463
    ##      6        1.3524             nan     0.1000    0.0427
    ##      7        1.3246             nan     0.1000    0.0361
    ##      8        1.3010             nan     0.1000    0.0358
    ##      9        1.2785             nan     0.1000    0.0280
    ##     10        1.2585             nan     0.1000    0.0324
    ##     20        1.1003             nan     0.1000    0.0206
    ##     40        0.9238             nan     0.1000    0.0094
    ##     60        0.8167             nan     0.1000    0.0092
    ##     80        0.7345             nan     0.1000    0.0048
    ##    100        0.6704             nan     0.1000    0.0042
    ##    120        0.6178             nan     0.1000    0.0030
    ##    140        0.5741             nan     0.1000    0.0024
    ##    150        0.5547             nan     0.1000    0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1886
    ##      2        1.4888             nan     0.1000    0.1251
    ##      3        1.4071             nan     0.1000    0.1106
    ##      4        1.3356             nan     0.1000    0.0819
    ##      5        1.2834             nan     0.1000    0.0739
    ##      6        1.2371             nan     0.1000    0.0578
    ##      7        1.1995             nan     0.1000    0.0648
    ##      8        1.1591             nan     0.1000    0.0481
    ##      9        1.1272             nan     0.1000    0.0448
    ##     10        1.0985             nan     0.1000    0.0474
    ##     20        0.8903             nan     0.1000    0.0212
    ##     40        0.6753             nan     0.1000    0.0129
    ##     60        0.5431             nan     0.1000    0.0083
    ##     80        0.4520             nan     0.1000    0.0046
    ##    100        0.3862             nan     0.1000    0.0038
    ##    120        0.3347             nan     0.1000    0.0033
    ##    140        0.2915             nan     0.1000    0.0018
    ##    150        0.2732             nan     0.1000    0.0015
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2407
    ##      2        1.4587             nan     0.1000    0.1687
    ##      3        1.3516             nan     0.1000    0.1172
    ##      4        1.2763             nan     0.1000    0.1028
    ##      5        1.2113             nan     0.1000    0.0936
    ##      6        1.1527             nan     0.1000    0.0861
    ##      7        1.0998             nan     0.1000    0.0699
    ##      8        1.0560             nan     0.1000    0.0604
    ##      9        1.0180             nan     0.1000    0.0538
    ##     10        0.9837             nan     0.1000    0.0461
    ##     20        0.7475             nan     0.1000    0.0298
    ##     40        0.5190             nan     0.1000    0.0119
    ##     60        0.3897             nan     0.1000    0.0057
    ##     80        0.3091             nan     0.1000    0.0047
    ##    100        0.2497             nan     0.1000    0.0030
    ##    120        0.2086             nan     0.1000    0.0022
    ##    140        0.1771             nan     0.1000    0.0018
    ##    150        0.1646             nan     0.1000    0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1308
    ##      2        1.5240             nan     0.1000    0.0884
    ##      3        1.4651             nan     0.1000    0.0654
    ##      4        1.4200             nan     0.1000    0.0522
    ##      5        1.3854             nan     0.1000    0.0433
    ##      6        1.3567             nan     0.1000    0.0459
    ##      7        1.3266             nan     0.1000    0.0372
    ##      8        1.3020             nan     0.1000    0.0348
    ##      9        1.2794             nan     0.1000    0.0365
    ##     10        1.2574             nan     0.1000    0.0330
    ##     20        1.1017             nan     0.1000    0.0195
    ##     40        0.9260             nan     0.1000    0.0089
    ##     60        0.8220             nan     0.1000    0.0082
    ##     80        0.7426             nan     0.1000    0.0044
    ##    100        0.6791             nan     0.1000    0.0036
    ##    120        0.6264             nan     0.1000    0.0036
    ##    140        0.5817             nan     0.1000    0.0023
    ##    150        0.5611             nan     0.1000    0.0026
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1807
    ##      2        1.4920             nan     0.1000    0.1359
    ##      3        1.4063             nan     0.1000    0.1053
    ##      4        1.3395             nan     0.1000    0.0830
    ##      5        1.2872             nan     0.1000    0.0740
    ##      6        1.2405             nan     0.1000    0.0729
    ##      7        1.1954             nan     0.1000    0.0635
    ##      8        1.1561             nan     0.1000    0.0512
    ##      9        1.1233             nan     0.1000    0.0437
    ##     10        1.0954             nan     0.1000    0.0473
    ##     20        0.8898             nan     0.1000    0.0201
    ##     40        0.6790             nan     0.1000    0.0109
    ##     60        0.5492             nan     0.1000    0.0051
    ##     80        0.4593             nan     0.1000    0.0075
    ##    100        0.3914             nan     0.1000    0.0062
    ##    120        0.3397             nan     0.1000    0.0031
    ##    140        0.2984             nan     0.1000    0.0027
    ##    150        0.2796             nan     0.1000    0.0022
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2371
    ##      2        1.4602             nan     0.1000    0.1662
    ##      3        1.3555             nan     0.1000    0.1303
    ##      4        1.2731             nan     0.1000    0.0964
    ##      5        1.2108             nan     0.1000    0.0928
    ##      6        1.1531             nan     0.1000    0.0710
    ##      7        1.1068             nan     0.1000    0.0695
    ##      8        1.0627             nan     0.1000    0.0568
    ##      9        1.0271             nan     0.1000    0.0560
    ##     10        0.9914             nan     0.1000    0.0597
    ##     20        0.7598             nan     0.1000    0.0260
    ##     40        0.5332             nan     0.1000    0.0160
    ##     60        0.3981             nan     0.1000    0.0054
    ##     80        0.3186             nan     0.1000    0.0075
    ##    100        0.2598             nan     0.1000    0.0034
    ##    120        0.2173             nan     0.1000    0.0028
    ##    140        0.1820             nan     0.1000    0.0019
    ##    150        0.1675             nan     0.1000    0.0010
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1264
    ##      2        1.5250             nan     0.1000    0.0873
    ##      3        1.4684             nan     0.1000    0.0651
    ##      4        1.4250             nan     0.1000    0.0505
    ##      5        1.3914             nan     0.1000    0.0473
    ##      6        1.3603             nan     0.1000    0.0461
    ##      7        1.3308             nan     0.1000    0.0335
    ##      8        1.3094             nan     0.1000    0.0369
    ##      9        1.2862             nan     0.1000    0.0316
    ##     10        1.2668             nan     0.1000    0.0328
    ##     20        1.1100             nan     0.1000    0.0172
    ##     40        0.9386             nan     0.1000    0.0099
    ##     60        0.8280             nan     0.1000    0.0063
    ##     80        0.7459             nan     0.1000    0.0041
    ##    100        0.6824             nan     0.1000    0.0042
    ##    120        0.6295             nan     0.1000    0.0025
    ##    140        0.5857             nan     0.1000    0.0030
    ##    150        0.5656             nan     0.1000    0.0023
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.1901
    ##      2        1.4885             nan     0.1000    0.1240
    ##      3        1.4073             nan     0.1000    0.1034
    ##      4        1.3420             nan     0.1000    0.0855
    ##      5        1.2863             nan     0.1000    0.0733
    ##      6        1.2401             nan     0.1000    0.0680
    ##      7        1.1972             nan     0.1000    0.0522
    ##      8        1.1638             nan     0.1000    0.0555
    ##      9        1.1292             nan     0.1000    0.0487
    ##     10        1.0978             nan     0.1000    0.0420
    ##     20        0.8999             nan     0.1000    0.0266
    ##     40        0.6816             nan     0.1000    0.0095
    ##     60        0.5552             nan     0.1000    0.0104
    ##     80        0.4643             nan     0.1000    0.0069
    ##    100        0.3949             nan     0.1000    0.0049
    ##    120        0.3428             nan     0.1000    0.0032
    ##    140        0.3001             nan     0.1000    0.0027
    ##    150        0.2800             nan     0.1000    0.0020
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2325
    ##      2        1.4616             nan     0.1000    0.1648
    ##      3        1.3583             nan     0.1000    0.1227
    ##      4        1.2806             nan     0.1000    0.1151
    ##      5        1.2107             nan     0.1000    0.0937
    ##      6        1.1515             nan     0.1000    0.0758
    ##      7        1.1029             nan     0.1000    0.0707
    ##      8        1.0582             nan     0.1000    0.0627
    ##      9        1.0180             nan     0.1000    0.0468
    ##     10        0.9873             nan     0.1000    0.0592
    ##     20        0.7489             nan     0.1000    0.0321
    ##     40        0.5210             nan     0.1000    0.0107
    ##     60        0.3978             nan     0.1000    0.0063
    ##     80        0.3166             nan     0.1000    0.0032
    ##    100        0.2571             nan     0.1000    0.0031
    ##    120        0.2150             nan     0.1000    0.0027
    ##    140        0.1803             nan     0.1000    0.0023
    ##    150        0.1660             nan     0.1000    0.0017
    ## 
    ## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
    ##      1        1.6094             nan     0.1000    0.2383
    ##      2        1.4578             nan     0.1000    0.1546
    ##      3        1.3561             nan     0.1000    0.1220
    ##      4        1.2775             nan     0.1000    0.1062
    ##      5        1.2117             nan     0.1000    0.0826
    ##      6        1.1591             nan     0.1000    0.0760
    ##      7        1.1112             nan     0.1000    0.0715
    ##      8        1.0659             nan     0.1000    0.0554
    ##      9        1.0301             nan     0.1000    0.0561
    ##     10        0.9942             nan     0.1000    0.0528
    ##     20        0.7583             nan     0.1000    0.0253
    ##     40        0.5369             nan     0.1000    0.0110
    ##     60        0.4095             nan     0.1000    0.0088
    ##     80        0.3275             nan     0.1000    0.0055
    ##    100        0.2670             nan     0.1000    0.0021
    ##    120        0.2246             nan     0.1000    0.0023
    ##    140        0.1887             nan     0.1000    0.0010
    ##    150        0.1753             nan     0.1000    0.0019

``` r
saveRDS(mod_gbm, "mod_gbm.rds")
mod_gbm <- readRDS("mod_gbm.rds")

# Linear Discriminant Analysis
mod_lda <- train(classe ~., data = training_set, method = "lda")
saveRDS(mod_lda, "mod_lda.rds")
mod_lda <- readRDS("mod_lda.rds")
```

In order to effectively compare how individual models perform compared
with the ensemble, I look at the accuracy of each model as shown below:

``` r
pred_rf <- predict(mod_rf, newdata = validation_set)
pred_gbm <- predict(mod_gbm, newdata = validation_set)
pred_lda <- predict(mod_lda, newdata = validation_set)

confusionMatrix(pred_rf, validation_set$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2231    8    0    0    0
    ##          B    1 1502    9    0    0
    ##          C    0    8 1353   26    2
    ##          D    0    0    6 1260    5
    ##          E    0    0    0    0 1435
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9917          
    ##                  95% CI : (0.9895, 0.9936)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9895          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9996   0.9895   0.9890   0.9798   0.9951
    ## Specificity            0.9986   0.9984   0.9944   0.9983   1.0000
    ## Pos Pred Value         0.9964   0.9934   0.9741   0.9913   1.0000
    ## Neg Pred Value         0.9998   0.9975   0.9977   0.9960   0.9989
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1914   0.1724   0.1606   0.1829
    ## Detection Prevalence   0.2854   0.1927   0.1770   0.1620   0.1829
    ## Balanced Accuracy      0.9991   0.9939   0.9917   0.9891   0.9976

``` r
confusionMatrix(pred_gbm, validation_set$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2203   46    0    4    1
    ##          B   22 1428   38    8   19
    ##          C    3   41 1308   41    4
    ##          D    2    2   19 1225   19
    ##          E    2    1    3    8 1399
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9639          
    ##                  95% CI : (0.9596, 0.9679)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9544          
    ##                                           
    ##  Mcnemar's Test P-Value : 2.108e-06       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9870   0.9407   0.9561   0.9526   0.9702
    ## Specificity            0.9909   0.9863   0.9863   0.9936   0.9978
    ## Pos Pred Value         0.9774   0.9426   0.9363   0.9669   0.9901
    ## Neg Pred Value         0.9948   0.9858   0.9907   0.9907   0.9933
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2808   0.1820   0.1667   0.1561   0.1783
    ## Detection Prevalence   0.2873   0.1931   0.1781   0.1615   0.1801
    ## Balanced Accuracy      0.9890   0.9635   0.9712   0.9731   0.9840

``` r
confusionMatrix(pred_lda, validation_set$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1792  231  115   71   43
    ##          B   48  960  131   58  259
    ##          C  211  181  925  140  114
    ##          D  177   67  166  971  156
    ##          E    4   79   31   46  870
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.7033         
    ##                  95% CI : (0.693, 0.7134)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.625          
    ##                                          
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8029   0.6324   0.6762   0.7551   0.6033
    ## Specificity            0.9181   0.9216   0.9003   0.9137   0.9750
    ## Pos Pred Value         0.7957   0.6593   0.5888   0.6318   0.8447
    ## Neg Pred Value         0.9213   0.9127   0.9294   0.9501   0.9161
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2284   0.1224   0.1179   0.1238   0.1109
    ## Detection Prevalence   0.2870   0.1856   0.2002   0.1959   0.1313
    ## Balanced Accuracy      0.8605   0.7770   0.7882   0.8344   0.7892

``` r
confusionMatrix(pred_rf, validation_set$classe)$overall[1]
```

    ##  Accuracy 
    ## 0.9917155

``` r
confusionMatrix(pred_gbm, validation_set$classe)$overall[1]
```

    ##  Accuracy 
    ## 0.9639307

``` r
confusionMatrix(pred_lda, validation_set$classe)$overall[1]
```

    ##  Accuracy 
    ## 0.7032883

I then generate a level-one dataset for training the ensemble
metalearner, train the ensemble metalearner with the generated level-one
training dataset, and save the stack model to an rds file. Again, so
that I would not have to run it again just in case.

``` r
# Generate a level-one dataset for training the ensemble metalearner
predDF <- data.frame(pred_rf, pred_gbm, pred_lda, classe = validation_set$classe, stringsAsFactors = F)

# Train the ensemble
set.seed(100)
modelStack <- train(classe ~ ., data = predDF, method = "rf")
saveRDS(modelStack, "modelStack.rds")
modelStack <- readRDS("modelStack.rds")
```

To compare how the ensemble metalearner pans with the predictive power
of individual models, I look at the accuracy of the predicted classes of
the metalearner. So it looks like the ensemble metalearner’s predictive
power is at par with the random forest. This makes sense tho, because
the accuracy using random forest is already high.

``` r
modelPred <- predict(modelStack, newdata = predDF)
confusionMatrix(modelPred, predDF$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2231    8    0    0    0
    ##          B    1 1502    9    0    0
    ##          C    0    8 1353   26    2
    ##          D    0    0    6 1260    5
    ##          E    0    0    0    0 1435
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9917          
    ##                  95% CI : (0.9895, 0.9936)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9895          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9996   0.9895   0.9890   0.9798   0.9951
    ## Specificity            0.9986   0.9984   0.9944   0.9983   1.0000
    ## Pos Pred Value         0.9964   0.9934   0.9741   0.9913   1.0000
    ## Neg Pred Value         0.9998   0.9975   0.9977   0.9960   0.9989
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1914   0.1724   0.1606   0.1829
    ## Detection Prevalence   0.2854   0.1927   0.1770   0.1620   0.1829
    ## Balanced Accuracy      0.9991   0.9939   0.9917   0.9891   0.9976

``` r
confusionMatrix(modelPred, predDF$classe)$overall[1]
```

    ##  Accuracy 
    ## 0.9917155

Next, I generate predictions on the test set. Note that I’m using the
same names because R has this weird rule that the each column name
should be exactly the same as what was used in training the ensemble
model.

``` r
pred_rf <- predict(mod_rf, newdata = testing)
pred_gbm <- predict(mod_gbm, newdata = testing)
pred_lda <- predict(mod_lda, newdata = testing)

testPredLevelOne <- data.frame(pred_rf, pred_gbm, pred_lda, stringsAsFactors = F)
```

Now I’m ready to predict\!

## Predicting Using the Trained Ensemble Model

``` r
for (j in 1:20) {
  p <- predict(modelStack, testPredLevelOne[j,])
  print(p)
}
```

    ## [1] B
    ## Levels: A B C D E
    ## [1] A
    ## Levels: A B C D E
    ## [1] B
    ## Levels: A B C D E
    ## [1] A
    ## Levels: A B C D E
    ## [1] A
    ## Levels: A B C D E
    ## [1] E
    ## Levels: A B C D E
    ## [1] D
    ## Levels: A B C D E
    ## [1] B
    ## Levels: A B C D E
    ## [1] A
    ## Levels: A B C D E
    ## [1] A
    ## Levels: A B C D E
    ## [1] B
    ## Levels: A B C D E
    ## [1] C
    ## Levels: A B C D E
    ## [1] B
    ## Levels: A B C D E
    ## [1] A
    ## Levels: A B C D E
    ## [1] E
    ## Levels: A B C D E
    ## [1] E
    ## Levels: A B C D E
    ## [1] A
    ## Levels: A B C D E
    ## [1] B
    ## Levels: A B C D E
    ## [1] B
    ## Levels: A B C D E
    ## [1] B
    ## Levels: A B C D E
