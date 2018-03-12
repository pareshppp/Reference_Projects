# Modeling Strength of Concrete using ANNs
setwd(paste0("~/Code/Data Science Projects/",
             "Machine-Learning-with-R---Brett-Lantz/",
             "Modeling Strength of Concrete using ANNs"))

# Reading and exploring data
concrete <- read.csv('concrete.csv')
str(concrete)

# Neural networks work best when the input data are scaled to a narrow 
# range around zero
normalize <- function(x){
    (x - min(x)) / (max(x) - min(x))
}
concrete.norm <- as.data.frame(lapply(X = concrete, FUN = normalize))

summary(concrete$strength)
summary(concrete.norm$strength)


# train - test split :: 75:25 
concrete.norm.train <- concrete.norm[1:773, ]
concrete.norm.test <- concrete.norm[774:1030, ]


# training a model on the data
library(neuralnet)
concrete.model <- neuralnet(formula = strength ~ cement + slag + ash + water + 
                                superplastic + coarseagg + fineagg + age, 
                            data = concrete.norm.train)

# visualize model
plot(concrete.model)


# evaluating model performance
model.result <- compute(x = concrete.model, covariate = concrete.norm.test[, 1:8])
strength.pred <- model.result$net.result

cor(strength.pred, concrete.norm.test$strength)


# improving model performance
concrete.model2 <- neuralnet(formula = strength ~ cement + slag + ash + water + 
                                superplastic + coarseagg + fineagg + age, 
                            data = concrete.norm.train, hidden = 4)

plot(concrete.model2)

model.result2 <- compute(x = concrete.model2, covariate = concrete.norm.test[, 1:8])
strength.pred2 <- model.result2$net.result

cor(strength.pred2, concrete.norm.test$strength)
