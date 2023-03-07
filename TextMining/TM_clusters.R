library(httr)
library(jsonlite)
#install.packages('tidyverse')
library(tidyverse)
#install.packages('cluster')
library(cluster)
#install.packages('factoextra')
library(factoextra)
#install.packages('XML')
library(XML)
#install.packages('rvest')
library(rvest)
#install.packages('xml2')
library(xml2)

#You will need your own key to run this. 
#get one and the documentation here:https://api-ninjas.com/api/planets
api_key <- readLines("C:\\Users\\gameb\\OneDrive\\Desktop\\API Keys\\api_planets.txt")
base_url <- 'https://api.api-ninjas.com/v1/planets?'
###############################################################################
# Originally I did numerous api calls, but I narrowed my search down.         #
###############################################################################
#I tried putting this part into a lapply function but it was too finicky. 
#Hard coded works, and isn't too long.
# options1 <- c('min_temperature'='200', 'max_temperature'='269')
# options2 <- c('min_temperature'='270', 'max_temperature'='339')
# options3 <- c('min_temperature'='340', 'max_temperature'='409')
# #making the query options in the right format
# query_options1 <- paste(names(options1), options1, sep='=', collapse = '&')
# query_options2 <- paste(names(options2), options2, sep='=', collapse = '&')
# query_options3 <- paste(names(options3), options3, sep='=', collapse = '&')
# queries <- c(query_options1,query_options2,query_options3)
# #creating the full query url
# lapply(queries, function(x) {
#   temp <- paste0(base_url,x) 
# }) -> urls
# urls <- do.call(rbind, urls) #makes it look a little nicer
# lapply(urls, function(x) {
#   geturl <- httr::GET(x, add_headers('X-Api-Key'=api_key)) 
#   # content(geturl,"text", encoding ="UTF-8")
#   api_char <- base::rawToChar(geturl$content)
#   api_json <- jsonlite::fromJSON(api_char, flatten=TRUE)
# }) -> original_df #I'm doing 3 api calls because there is a limit of 30 results.
# original_df <- do.call(rbind, original_df) #collate all results into one dataframe
#I believe there is an option to put it in pages but this worked
###############################################################################


#This is around the range of habitable average temperatures for us (in Kelvin)
#Earth is 288 Kelvin for reference.
options <- c('min_temperature'='263', 'max_temperature'='313')
query_options <- paste(names(options), options, sep='=', collapse = '&')
url <- paste0(base_url, query_options)
geturl <- httr::GET(url, add_headers('X-Api-Key'=api_key))
api_char <- base::rawToChar(geturl$content)
original_df <- jsonlite::fromJSON(api_char, flatten=TRUE)


#Cleaning the data
df <- original_df #work on a copy
rownames(df) <- df[,1] #setting labels as the name of the planet
#getting rid of these columns, because they have the most NA values
df <- df[,!names(df) %in% c("mass", "radius", "name")] 
#this is a major outlier in the data, so I 
# df <- df[!(row.names(df) %in% "CFBDSIR J145829+101343 b"),] 
df <- na.omit(df) #get rid of na's for clustering
df <- scale(df) #scale the values
#Seeing the difference in scaled vs unscaled
#head(original_df)
#head(df)

#we needed to get rid of NA values for this
fviz_nbclust(df, kmeans, method="silhouette")

# num_clusters <- c(2:6)
# lapply(num_clusters, function(x) {
#   kx <- kmeans(df, centers=x, nstart=25)
#   fviz_cluster(kx, data=df)
# })
k2 <- kmeans(df, centers=2, nstart=25)
fviz_cluster(k2, data=df)

#While this was an alright use of the api, I think the amount of data I obtained
#was lackluster. So, I'm doing more work but with a different set of data.



###############################################################################
# Using Data from upr.edu (ish)                                               #
###############################################################################

# page <- read_html('https://phl.upr.edu/projects/habitable-exoplanets-catalog')
# stats <- page %>%
#   html_element(css ="#h\\.15ceb14a6001e5d2_2006 > div > div > div > div > div.YMEQtf") %>%
#   html_table()
# stats
#Originally I wanted to use this site, but the web scraping wasn't working so I used Wikipedia instead
#Wikipedia does use this a a primary source, so it isn't terrible
page <- read_html('https://en.wikipedia.org/wiki/List_of_potentially_habitable_exoplanets')
#page

stats <- page %>%
  html_element(css ='#mw-content-text > div.mw-parser-output > table') %>%
  html_table()
stats
#colnames(stats)
web <- stats #work on a copy


#getting rid of these columns, because they have the most NA values
cols_to_del <- c("Star", "Star type", "Mass (M⊕)", "Radius (R⊕)", "Density (g/cm3)", "Refs/Notes")
web <- web[,!names(web) %in% cols_to_del] 
#this is a major outlier in the data, so I got rid of id
web <- web[web$Object != "L 98-59 f",] 
#removed some empty values here
web <- web[web$`Teq (K)` != "",] 
#rename rows (mainly don't want the weird symbol)
new_col_names <- c("Name", "Flux", "Temperature", "Period", "Distance")
names(web) <- new_col_names 
web <- transform(web, Flux = as.numeric(Flux), Temperature = as.numeric(Temperature))
rownames(web) <- web[,1] #I can set labels now that it's a dataframe
web <- web[,!names(web) %in% c("Name")]
#Last bit of cleaning before k means
web <- na.omit(web) #get rid of na's for clustering
web <- scale(web) #scale the values

#Using the silhouette method we can find a good number of clusters
fviz_nbclust(web, kmeans, method="silhouette")
#Look at each clustering from 2 to 6 clusters
num_clusters <- c(2:6)
lapply(num_clusters, function(x) {
  kx <- kmeans(web, centers=x, nstart=25)
  fviz_cluster(kx, data=web)
})
#I would agree with the silhouette graph that 4 clusters is a good number,
#but I think that 3 clusters would also does just as well.
k3 <- kmeans(web, centers=3, nstart=25)
fviz_cluster(k3, data=web)





###############################################################################
# Hierarchical clustering with this data                                      #
###############################################################################


web.dist <- dist(web)
hc.out_web <- hclust(web.dist, method="ward.D")
plot(hc.out_web, main = "Cluster Dendrogram of Euclidian Distance")
rect.hclust(hc.out_web, k=3, border = 2:5)


#After doing that I compared it to cosine similarity.
web2 <- t(web) #transpose to cluster the things I want.
cosine_dist = 1-crossprod(web2) /(sqrt(colSums(web2^2)%*%t(colSums(web2^2))))

# # remove NaN's by 0 if needed
# cosine_dist[is.na(cosine_dist)] <- 0

# create dist object
cosine_dist <- as.dist(cosine_dist)
cluster <- hclust(cosine_dist, method = "ward.D")
plot(cluster, main = "Cluster Dendrogram of Cosine Similarity")
rect.hclust(cluster, k=3, border = 2:5)








