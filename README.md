# Practical Machine Learning

The goal of this machine learning project is to predict the manner in which participants did their exercise. This refers to the "classe" variable in the training set. Finally, the  machine learning algorithm is used on 20 test cases available in the test data to predict the classes given only the features in the dataset.

# Experimental Design

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Files

This repository contains the following files:

- `CodeBook.Rmd` - a code book that describes the variables, the data, and any transformations or work that I performed to clean up the data

- `Final_Project.Rmd` - a descriptive code used to clean, mangle and predict outcomes

- `Final_Project.md` - md version of the Rmd file so that it is easier to view on github
  
- `Final_Project.html` - HTML file generated from the Rmd code
  
- `mod_gbm.rds, mod_lda.rds, mod_rf.rds, modelStack.rds` - models for gradient boost, linear discriminant analysis, random forest, and the metalearner stored as rds files for easy access.