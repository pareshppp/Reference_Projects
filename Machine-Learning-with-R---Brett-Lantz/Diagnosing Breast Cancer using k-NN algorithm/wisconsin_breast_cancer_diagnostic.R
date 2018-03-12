## Breast Cancer Wisconsin (Diagnostic) Data Set with R

setwd("~/Code/R/R Projects/Wisconsin Breast Cancer/Breast_Cancer_Wisconsin")

# library(dplyr)

# reading data
wdbc <- read.csv('wdbc-data.csv', stringsAsFactors = F)
colnames(wdbc) <- c('id', 'diagnosis', 
                 'radius.mean', 'texture.mean', 'perimeter.mean', 'area.mean', 
                 'smoothness.mean', 'compactness.mean', 'concavity.mean', 
                 'concave_points.mean', 'symmetry.mean', 'fractal_dimension.mean', 
                 'radius.sd', 'texture.sd', 'perimeter.sd', 'area.sd', 
                 'smoothness.sd', 'compactness.sd', 'concavity.sd', 
                 'concave_points.sd', 'symmetry.sd', 'fractal_dimension.sd',
                 'radius.worst', 'texture.worst', 'perimeter.worst', 'area.worst', 
                 'smoothness.worst', 'compactness.worst', 'concavity.worst', 
                 'concave_points.worst', 'symmetry.worst', 'fractal_dimension.worst')

# Adding Labels to diagnosis
wdbc$diagnosis <- factor(x = wdbc$diagnosis, levels = c('B', 'M'), 
                         labels = c('Benign', 'Malignant'))

# creating original copy
wdbc.org <- wdbc

# checking data
str(wdbc)
table(wdbc$diagnosis)
prop.table(table(wdbc$diagnosis))

# don't need id for prediction
wdbc <- wdbc[-1]

# normalizing numeric data
normalize <- function(x){
    (x - min(x)) / (max(x) - min(x))
}

wdbc.norm <- as.data.frame(lapply(X = wdbc[, 2:31], FUN = normalize))

# splitting train and test
wdbc.train <- wdbc.norm[1:469, ]
wdbc.test <- wdbc.norm[470:568, ]

# saving diagnosis separately
wdbc.train.labels <- wdbc[1:469, 1]
wdbc.test.labels <- wdbc[470:568, 1]

# loading class package for k-NN
library(class)

# predicting using knn() in class package
# sice we have 469 instances, taking k=21, odd number root
wdbc.predict <- knn(train = wdbc.train, test = wdbc.test, 
                    cl = wdbc.train.labels, k = 21)

# Evaluating prediction
library(gmodels)
CrossTable(x = wdbc.test.labels, y = wdbc.predict, prop.chisq = F)


#################################################################
# Improving model performance
#################################################################

# z-score standardisation
wdbc.ztrans <- as.data.frame(scale(x = wdbc[-1]))

summary(wdbc.ztrans)

# Doing the same again
wdbc.train <- wdbc.ztrans[1:469, ]
wdbc.test <- wdbc.ztrans[470:568, ]

wdbc.predict <- knn(train = wdbc.train, test = wdbc.test, 
                    cl = wdbc.train.labels, k = 21)

CrossTable(x = wdbc.test.labels, y = wdbc.predict, prop.chisq = F)

################################################################

# Testing alternative values of k