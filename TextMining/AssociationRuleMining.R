# install.packages("arules")
# install.packages("arulesViz")
library(arules)
library(arulesViz)

#Data downloaded from the NASA Exoplanet Archive.
#It could have been possible to webscrape, but there is a ton of unecessary data there
#https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS
original_df <- read.csv("https://raw.githubusercontent.com/msolko/Spring2023/main/MachineLearning/Planet_Data.csv")
df <- original_df #work on a copy
df <- df[,!names(df) %in% c("hostname", "default_flag")] 
df <- na.omit(df) #get rid of na's 
#summary(df) #check data
df <- discretizeDF(df) #discretize values to work in ARM
df[] <- lapply(df, factor)
rule1 <- apriori(df, parameter = list(support=0.15, confidence=0.6, minlen=3, maxlen = 4))

top_support <- sort(rule1, decreasing = TRUE, na.last = NA, by = "support")
inspect(head(top_support, 15))
top_conf <- sort(rule1, decreasing = TRUE, na.last = NA, by = "confidence")
inspect(head(top_conf, 15))
top_lift <- sort(rule1, decreasing = TRUE, na.last = NA, by = "lift")
inspect(head(top_lift, 15))


plot(rule1)
plot(rule1, method='grouped')

















