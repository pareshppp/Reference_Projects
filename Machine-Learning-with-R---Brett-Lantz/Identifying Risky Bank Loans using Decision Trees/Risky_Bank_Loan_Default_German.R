# Identifying Risky Bank Loans
setwd(paste0("~/Code/R/R Projects/Risky Bank Loan Default ",
             "(German Credit data - Hans Hoffman)",
             "/Risky_Bank_Loan_Default_German"))

# Reading, checking and preparing data
credit <- read.csv('credit.csv')
str(credit)

table(credit$checking_balance)
table(credit$savings_balance)

summary(credit$months_loan_duration)
summary(credit$amount)

credit$default <- factor(x = credit$default, labels = c('No', 'Yes'))

table(credit$default)



# Spliting train and test
set.seed(123)

train.sample <- sample(x = 1000, size = 900, replace = F)

credit.train <- credit[train.sample, ]
credit.test <- credit[-train.sample, ]

prop.table(table(credit.train$default))
prop.table(table(credit.test$default))


# training a model on the data
library(C50)

credit.model <- C5.0(x = credit.train[-17], y = credit.train$default)

summary(credit.model)


# evaluating model performance
credit.predict <- predict(object = credit.model, newdata = credit.test)

library(gmodels)
CrossTable(x = credit.test$default, y = credit.predict, 
           prop.r = F, prop.c = F, prop.chisq = F, 
           dnn = c('actual default', 'predicted default'))


# improving model performance

# Boosting the accuracy of decision trees
credit.model.boost10 <- C5.0(x = credit.train[-17], 
                             y = credit.train$default, trials = 10)

summary(credit.model.boost10)

credit.predict.boost10 <- predict(object = credit.model.boost10, 
                                  newdata = credit.test)
CrossTable(x = credit.test$default, y = credit.predict.boost10, 
           prop.r = F, prop.c = F, prop.chisq = F, 
           dnn = c('actual default', 'predicted default'))


# Making mistakes more costlier than others
# creating cost matrix
matrix.dim <- list(c('No', 'Yes'), c('No', 'Yes'))
names(matrix.dim) <- c('Predicted', 'Actual')

# Suppose we believe that a loan default costs the bank four times as much
# as a missed opportunity. Our penalty values could then be defined as:
error.cost <- matrix(data = c(0,1,4,0), nrow = 2, dimnames = matrix.dim)

# predicting
credit.model.cost <- C5.0(x = credit.train[-17], 
                             y = credit.train$default, costs = error.cost)

credit.predict.cost <- predict(object = credit.model.cost, 
                                  newdata = credit.test)
CrossTable(x = credit.test$default, y = credit.predict.cost, 
           prop.r = F, prop.c = F, prop.chisq = F, 
           dnn = c('actual default', 'predicted default'))

# reduction of false negatives at the expense of increasing false 
# positives may be acceptable if our cost estimates were accurate
