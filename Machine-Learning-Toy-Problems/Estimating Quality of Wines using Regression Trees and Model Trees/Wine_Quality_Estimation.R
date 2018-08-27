# Estimating Quality of Wines using Regression Trees and Model Trees
setwd(paste0("~/Code/Data Science Projects/",
          "Machine-Learning-with-R---Brett-Lantz/",
          "Estimating Quality of Wines using Regression Trees and Model Trees"))

# Reading and exploring data
wine <- read.csv('whitewines.csv')

hist(wine$quality)

# splitting train & test - 75:25
wine.train <- wine[1:3750, ]
wine.test <- wine[3751:4898, ]


# training a model on the data
library(rpart)
model.rpart <- rpart(formula = quality ~ ., data = wine.train)
model.rpart
summary(model.rpart)

# Visualizing decision trees
library(rpart.plot)
rpart.plot(x = model.rpart, digits = 3)


# evaluating model performance
predict.rpart <- predict(object = model.rpart, newdata = wine.test)

# A quick look at the summary statistics of our predictions suggests 
# a potential problem; the predictions fall on a much narrower range 
# than the true values
summary(predict.rpart)
summary(wine.test$quality)
# This finding suggests that the model is not correctly identifying the 
# extreme cases, in particular the best and worst wines. On the other 
# hand, between the first and third quartile, we may be doing well.

# We'll use corelation to compare how well the predicted values 
# correspond to the true values
cor(x = predict.rpart, y = wine.test$quality)


# Measuring performance with the mean absolute error
MAE <- function(actual, predicted){
  mean(abs(actual - predicted))
}

MAE(actual = wine.test$quality, predicted = predict.rpart)

# This implies that, on average, the difference between our model's 
# predictions and the true quality score was about 0.57. On a quality 
# scale from zero to 10, this seems to suggest that our model is 
# doing fairly well.


# On the other hand, recall that most wines were neither very good nor 
# very bad; the typical quality score was around five to six. Therefore,
# a classifier that did nothing but predict the mean value may still do 
# fairly well according to this metric.

mean(wine.test$quality)

# If we predicted the value 5.85 for every wine sample, we would have a 
# mean absolute error of only about
MAE(actual = 5.85, wine.test$quality)

# Our regression tree (MAE = 0.57) comes closer on average to the true 
# quality score than the imputed mean (MAE = 0.59), but not by much.



# improving model performance
library(RWeka)
model.m5p <- M5P(formula = quality ~., data = wine.train)
model.m5p
summary(model.m5p)

predict.m5p <- predict(object = model.m5p, newdata = wine.test)

cor(x = predict.m5p, y = wine.test$quality)
MAE(actual = wine.test$quality, predicted = predict.m5p)
