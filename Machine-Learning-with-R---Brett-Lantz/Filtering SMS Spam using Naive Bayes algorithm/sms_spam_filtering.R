# Filtering mobile phone spam with the Naive Bayes algorithm

setwd("~/Code/R/R Projects/SMS Spam Filtering/SMS_Spam_Filtering")

# Getting and viewing data
sms.raw <- read.csv('sms_spam.csv', stringsAsFactors = F)

str(sms.raw)

sms.raw$type <- as.factor(sms.raw$type)

str(sms.raw)
prop.table(table(sms.raw$type))

# Data preparation - cleaning and standardizing text data
library(tm)

# creating a corpus of text from sms_raw$text
sms.corpus <- VCorpus(VectorSource(x = sms.raw$text))
print(sms.corpus)

# inspecting the corpus
inspect(sms.corpus[1:2])

# view messages
as.character(sms.corpus[[1]])
lapply(sms.corpus[1:2], as.character)

# cleaning the corpus
# transforming to lowercase
sms.corpus.clean <- tm_map(x = sms.corpus, 
                           FUN = content_transformer(tolower))
# checking
as.character(sms.corpus[[1]]); as.character(sms.corpus.clean[[1]])

# Although some numbers may provide useful information, 
# the majority would likely be unique to individual senders and 
# thus will not provide useful patterns across all messages.
# With this in mind, we'll strip all the numbers from the corpus
sms.corpus.clean <- tm_map(x = sms.corpus.clean, FUN = removeNumbers)


# Our next task is to remove filler words such as to, and, but, and or 
# from our SMS messages. These terms are known as stop words and are 
# typically removed prior to text mining. This is due to the fact that 
# although they appear very frequently, they do not provide much 
# useful information for machine learning.
sms.corpus.clean <- tm_map(x = sms.corpus.clean, 
                           FUN = removeWords, stopwords())
sms.corpus.clean <- tm_map(sms.corpus.clean, FUN = removePunctuation)


# Another common standardization for text data involves reducing 
# words to their root form in a process called stemming. The stemming 
# process takes words like learned, learning, and learns, and strips 
# the suffix in order to transform them into the base form, learn.
# Use wordstem() from SnowballC package
# Using stemDocument() in tm_map
library(SnowballC)
sms.corpus.clean <- tm_map(sms.corpus.clean, FUN = stemDocument)

# After removing numbers, stop words, and punctuation as well as 
# performing stemming, the text messages are left with the blank spaces 
# that previously separated the now-missing pieces. The final step in 
# our text cleanup process is to remove additional whitespace, using 
# the built-in stripWhitespace() transformation
sms.corpus.clean <- tm_map(sms.corpus.clean, FUN = stripWhitespace)

# checking data
as.character(sms.corpus[[3]]); as.character(sms.corpus.clean[[3]])



# Data preparation – splitting text documents into words

# Now that the data are processed to our liking, the final step 
# is to split the messages into individual components through a 
# process called tokenization.
# Creating a DTM sparse matrix
sms.dtm <- DocumentTermMatrix(x = sms.corpus.clean)

# sms.dtm2 <- DocumentTermMatrix(x = sms.corpus.clean, 
#                                control = list(tolower = T,
#                                               removePunctuation = T,
#                                               removeNumbers = T,
#                                               stopwords = T,
#                                               stemDocument = T))



# Data preparation – creating training and test datasets
# splitting at 75 %
sms.dtm.train <- sms.dtm[1:4169, ]
sms.dtm.test <- sms.dtm[4170:5574, ]

# getting type data
sms.dtm.train.labels <- sms.raw[1:4169, ]$type
sms.dtm.test.labels <- sms.raw[4170:5574, ]$type

# comparing proportions
prop.table(table(sms.dtm.train.labels))
prop.table(table(sms.dtm.test.labels))




# Visualizing text data – word clouds
library(wordcloud)
wordcloud(words = sms.corpus.clean, min.freq = 50, random.order = F)

# separate wordclouds for spam and ham
spam <- subset(x = sms.raw, type == 'spam')
ham <- subset(x = sms.raw, type == 'ham')

wordcloud(words = spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(words = ham$text, max.words = 40, scale = c(3, 0.5))


# Data preparation – creating indicator features for frequent words

# It's unlikely that all of the 6500 features are useful for 
# classification.To reduce the number of features, we will eliminate 
# any word that appear in less than five SMS messages, or in less than 
# about 0.1 percent of the records in the training data.

# getting frequent words
sms.freq.words <- findFreqTerms(x = sms.dtm.train, lowfreq = 5)

# filtering out infrequent words
sms.dtm.train.freq <- sms.dtm.train[, sms.freq.words]
sms.dtm.test.freq <- sms.dtm.test[, sms.freq.words]


# the cells in the sparse matrix are numeric and measure the number of 
# times a word appears in a message. We need to change this to a
# categorical variable that simply indicates yes or no depending on 
# whether the word appears at all.
convertCounts <- function(x){
    x <- ifelse(x > 0, 'Yes', 'No')
}

sms.train <- apply(X = sms.dtm.train.freq, MARGIN = 2, 
                   FUN = convertCounts)
sms.test <- apply(X = sms.dtm.test.freq, MARGIN = 2, 
                  FUN = convertCounts)





# training a model on the data
library(e1071)

# training model
sms.model <- naiveBayes(x = sms.train, y = sms.dtm.train.labels)

# generate predictions
sms.predict <- predict(sms.model, newdata = sms.test)


# evaluating model performance
library(gmodels)
CrossTable(x = sms.predict, y = sms.dtm.test.labels, prop.chisq = F, 
           prop.t = F, dnn = c('predicted', 'actual'))


# improving model performance
# adding laplace estimator
sms.model2 <- naiveBayes(x = sms.train, y = sms.dtm.train.labels, 
                         laplace = 1)

sms.predict2 <- predict(sms.model2, newdata = sms.test)

CrossTable(x = sms.predict2, y = sms.dtm.test.labels, prop.chisq = F, 
           prop.t = F, dnn = c('predicted', 'actual'))
