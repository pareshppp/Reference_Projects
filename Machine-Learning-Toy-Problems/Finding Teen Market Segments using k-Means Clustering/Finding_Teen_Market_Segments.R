# Finding Teen Market Segments using k-Means Clustering
setwd(paste0("~/Code/Data Science Projects/",
             "Machine-Learning-with-R---Brett-Lantz/",
             "Finding Teen Market Segments using k-Means Clustering"))

# exploring and preparing the data
teens <- read.csv('snsdata.csv')
str(teens)

table(teens$gender, useNA = 'ifany')

summary(teens$age)
# it is unlikely that a 3 year old or a 106 year old is attending high 
# school. To ensure that these extreme values don't cause problems for 
# the analysis, we'll need to clean them up before moving on.

# A more reasonable range of ages for the high school students includes 
# those who are at least 13 years old and not yet 20 years old. Any age 
# value falling outside this range should be treated the same as missing
# data—we cannot trust the age provided.
teens$age <- ifelse(teens$age >= 13 & teens$age < 20, teens$age, NA)
summary(teens$age)



# Data preparation – dummy coding missing values
teens$female <- ifelse(teens$gender == 'F' & !is.na(teens$gender), 1, 0)
teens$no_gender <- ifelse(is.na(teens$gender), 1, 0)
table(teens$gender, useNA = 'ifany')
table(teens$female)
table(teens$no_gender)

# Data preparation – imputing the missing values
# Most people in a graduation cohort were born within a single calendar 
# year. If we can identify the typical age for each cohort, we would have 
# a fairly reasonable estimate of the age of a student in that graduation 
# year.
mean(teens$age, na.rm = T)
aggregate(data = teens, age ~ gradyear, FUN = mean, na.rm = T)

ave.age <- ave(teens$age, teens$gradyear, 
               FUN = function(x){mean(x, na.rm = T)})
teens$age <- ifelse(is.na(teens$age), ave.age, teens$age)
summary(teens$age)


# training a model on the data
# The kmeans() function requires a data frame containing only numeric 
# data and a parameter specifying the desired number of clusters.
interests <- teens[5:40]
interests.z <- as.data.frame(lapply(interests, scale))

# The teenage characters in movies are identified in terms of five
# stereotypes: a brain, an athlete, a basket case, a princess, and a 
# criminal. Given that these identities prevail throughout popular 
# teen fiction, five seems like a reasonable starting point for k.
set.seed(2345)
teen.clusters <- kmeans(x = interests.z, centers = 5)


# evaluating model performance
# One of the most basic ways to evaluate the utility of a set of clusters
# is to examine the number of examples falling in each of the groups. 
# If the groups are too large or too small, they are not likely to be
# very useful.
teen.clusters$size

# For a more in-depth look at the clusters, we can examine the 
# coordinates of the cluster centroids
teen.clusters$centers
# By examining whether the clusters fall above or below the mean level 
# for each interest category, we can begin to notice patterns that 
# distinguish the clusters from each other.


# improving model performance
# applying the clusters back onto the full dataset
teens$cluster <- teen.clusters$cluster

head(teens[c('cluster', 'age', 'gender', 'friends')])
aggregate(data = teens, age ~ cluster, mean)
aggregate(data = teens, female ~ cluster, mean)
aggregate(data = teens, friends ~ cluster, mean)

# On an average, Princesses have the most friends (41.4), followed by 
# Athletes (37.2) and Brains (32.6). On the low end are Criminals (30.5) 
# and Basket Cases (27.7). As with gender, the connection between a 
# teen's number of friends and their predicted cluster is remarkable, 
# given that we did not use the friendship data as an input to the
# clustering algorithm. Also interesting is the fact that the number 
# of friends seems to be related to the stereotype of each clusters' 
# high school popularity; the stereotypically popular groups tend to 
# have more friends.
