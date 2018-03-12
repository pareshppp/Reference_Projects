# Predicting Insurance Premium
setwd(paste0("~/Code/Data Science Projects/",
             "Machine-Learning-with-R---Brett-Lantz",
             "/Predicting Insurance Premium using Linear Regression"))

# reading, exploring and preparing data
insurance <- read.csv('insurance.csv', stringsAsFactors = T)

str(insurance)

# checking for normal distribution
# although linear regression does not strictly require a normally distributed 
# dependent variable, the model often fits better when this is true.
summary(insurance$charges)
# Because the mean value is greater than the median, this implies that the 
# distribution of insurance expenses is right-skewed.
hist(insurance$charges)

table(insurance$region)


# Exploring relationships among features
cor(insurance[c('age', 'bmi', 'children', 'charges')])


# Visualizing relationships among features – scatterplot matrix
pairs(insurance[c('age', 'bmi', 'children', 'charges')])

library(psych)
pairs.panels(insurance[c('age', 'bmi', 'children', 'charges')])


# training a model on the data
insurance.model <- lm(formula = charges ~ ., data = insurance)
insurance.model

# evaluating model performance
summary(insurance.model)



# improving model performance

# adding non-linear relationships
insurance$age2 <- insurance$age ^ 2

# converting a numeric variable to a binary indicator
insurance$bmi30 <- ifelse(insurance$bmi >= 30, 1, 0)

# adding interaction effects


# an improved regression model
# Added a non-linear term for age. Created an indicator for obesity
# specified an interaction between obesity and smoking
insurance.model2 <- lm(formula = charges ~ age + age2 + sex + children
                       + region + bmi30 * smoker, data = insurance)

summary(insurance.model2)
