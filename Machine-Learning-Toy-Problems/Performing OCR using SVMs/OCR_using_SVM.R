# Performing OCR using SVMs
# setwd(paste0("~/Code/Data Science Projects/",
#              "Machine-Learning-with-R---Brett-Lantz/",
#              "Performing OCR using SVMs"))

# Reading and exploring data
letters <- read.csv('letterdata.csv')
str(letters)

# train and test split - 80:20
letters.train <- letters[1:16000, ]
letters.test <- letters[16001:20000, ]

# training a model on the data
library(kernlab)
letters.model <- ksvm(letter ~ ., data = letters.train, 
                      kernel = 'vanilladot')
letters.model

# evaluating model performance
letters.predict <- predict(letters.model, letters.test)
head(letters.predict)

table(letters.predict, letters.test$letter)

agreement <- letters.predict == letters.test$letter
prop.table(table(agreement))


# improving model performance
library(kernlab)
letters.model.rbf <- ksvm(letter ~ ., data = letters.train, 
                      kernel = 'rbfdot')

letters.predict.rbf <- predict(letters.model.rbf, letters.test)

agreement.rbf <- letters.predict.rbf == letters.test$letter
prop.table(table(agreement.rbf))
