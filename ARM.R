library(arules)
library(arulesViz)

#load in dataset
data = read.csv("data.csv")

#create lists from purchase history column
data$PurchaseHistoryList = strsplit(as.character(data$Purchase.History), ",")

# covert the list to transactions
transactions = as(data$PurchaseHistoryList, "transactions")

#apply apriori in order to get rules
rules = apriori(transactions, parameter = list(support = 0.1, confidence = 0.7))

#look at the rules
inspect(rules)

#plot a graph opf the top 5 rules
plot(rules[1:5], method = "graph")



