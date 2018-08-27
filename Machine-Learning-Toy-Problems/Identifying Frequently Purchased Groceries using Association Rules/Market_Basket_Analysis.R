# Identifying Frequently Purchased Groceries using Association Rules
setwd(paste0("~/Code/Data Science Projects/",
             "Machine-Learning-with-R---Brett-Lantz/",
             "Identifying Frequently Purchased Groceries using Association Rules"))

# exploring and preparing the data
# Instead, we need a dataset that does not treat a transaction as a set
# of positions to be filled (or not filled) with specific items, but 
# rather as a market basket that either contains or does not contain each
# particular item.

# Data preparation – creating a sparse matrix for transaction data
library(arules)
groceries <- read.transactions('groceries.csv', sep = ',')

summary(groceries)

inspect(x = groceries[1:5])

# support level for the first three items in the grocery data
itemFrequency(groceries[, 1:3])

# Visualizing item support
itemFrequencyPlot(x = groceries, support = 0.1)
itemFrequencyPlot(x = groceries, topN = 10)


# Visualizing the transaction data – plotting the sparse matrix
image(groceries[1:50])
image(sample(groceries, size = 100))



# training a model on the data
apriori(groceries)

# One way to approach the problem of setting a minimum support threshold
# is to think about the smallest number of transactions you would need
# before you would consider a pattern interesting. For instance, you could 
# argue that if an item is purchased twice a day (about 60 times in a 
# month of data), it may be an interesting pattern. From there, it is 
# possible to calculate the support level needed to find only the rules 
# matching at least that many transactions. Since 60 out of 9,835 equals 
# 0.006, we'll try setting the support there first.

# We'll start with a confidence threshold of 0.25, which means that in 
# order to be included in the results, the rule has to be correct at least 
# 25 percent of the time. This will eliminate the most unreliable rules, 
# while allowing some room for us to modify behavior with targeted promotions.

# In addition to the minimum support and confidence parameters, it is 
# helpful to set minlen = 2 to eliminate rules that contain fewer than 
# two items. This prevents uninteresting rules from being created
# simply because the item is purchased frequently, for instance, 
# {} -> whole milk. This rule meets the minimum support and confidence 
# because whole milk is purchased in over 25 percent of the transactions, 
#but it isn't a very actionable insight.

groceries.rules <- apriori(data = groceries, 
          parameter = list(support = 0.006, confidence = 0.25, 
                           minlen = 2))
groceries.rules



# evaluating model performance
summary(groceries.rules)

inspect(groceries.rules[1:3])


# improving model performance

# Sorting the set of association rules
# the best five rules according to the lift statistic
inspect(sort(x = groceries.rules, by = 'lift') [1:5])

# Taking subsets of association rules
# the marketing team is excited about the possibilities of creating an 
# advertisement to promote berries, which are now in season. Before finalizing 
# the campaign, however, they ask you to investigate whether berries are often 
# purchased with other items. To answer this question, we'll need to find all 
# the rules that include berries in some form.
berry.rules <- subset(x = groceries.rules, items %in% 'berries')
inspect(berry.rules)


# Saving association rules to a file or data frame
write(x = groceries.rules, file = 'groceryrules.csv', sep = ',', 
      row.names = F)

groceries.rules.df <- as(groceries.rules, Class = 'data.frame')
