 # -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:47:47 2023

@author: Max Solko
"""
#Heavily inspired by code from Ami Gates


import requests  ## for getting data from a server
import re   ## for regular expressions
import pandas as pd    ## for dataframes and related
from pandas import DataFrame
# from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation 
# import sklearn
from sklearn.cluster import KMeans
from sklearn import preprocessing
# from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D
# from scipy.cluster.hierarchy import ward, dendrogram


#these are the topics I'm comparing. They will have some overlap
topics=["spacex", "exoplanet", "nasa"]


filename="NewHeadlines.csv"
MyFILE=open(filename,"w")
### Place the column names in - write to the first row
WriteThis="LABEL,Date,Source,Title,Headline\n"
MyFILE.write(WriteThis)
MyFILE.close()


endpoint="https://newsapi.org/v2/everything"
f = open("C:\\Users\\gameb\\OneDrive\\Desktop\\API Keys\\newsapi.txt", "r")
my_api_key =f.readline()
f.close()


for topic in topics:

    ## Dictionary Structure
    URLPost = {'apiKey':my_api_key,
               'q':topic,
               'language':"en",
               #"pageSize":20} #don't need 100 from each one
               }

    response=requests.get(endpoint, URLPost)
    print(response)
    jsontxt = response.json()
    print(jsontxt)
    #####################################################
    
    
    ## Open the file for append
    MyFILE=open(filename, "a")
    LABEL=topic
    for items in jsontxt["articles"]:
                  
        #Author=items["author"]
        #Author=str(Author)
        #Author=Author.replace(',', '')
        
        Source=items["source"]["id"]

        
        Date=items["publishedAt"]
        ##clean up the date
        NewDate=Date.split("T")
        Date=NewDate[0]

        
        ## CLEAN the Title
        ##----------------------------------------------------------
        ##Replace punctuation with space
        # Accept one or more copies of punctuation         
        # plus zero or more copies of a space
        # and replace it with a single space
        Title=items["title"]
        Title=str(Title)
        #print(Title)
        Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(' +', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(r'\"', ' ', str(Title), flags=re.IGNORECASE)

        Title=re.sub(r'[^a-zA-Z]', " ", str(Title), flags=re.VERBOSE)
        Title=Title.replace(',', '')
        Title=' '.join(Title.split())
        Title=re.sub("\n|\r", "", Title)
        ##----------------------------------------------------------
        
        Headline=items["description"]
        Headline=str(Headline)
        Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(' +', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
        ## Be sure there are no commas in the headlines or it will
        ## write poorly to a csv file....
        Headline=Headline.replace(',', '')
        Headline=' '.join(Headline.split())
        Headline=re.sub("\n|\r", "", Headline)
        
        ### AS AN OPTION - remove words of a given length............
        Headline = ' '.join([wd for wd in Headline.split() if len(wd)>3])
    
        
        WriteThis=str(LABEL)+","+str(Date)+","+str(Source)+","+ str(Title) + "," + str(Headline) + "\n"
        
        MyFILE.write(WriteThis)
        
    ## CLOSE THE FILE
    MyFILE.close()
    
################## END for loop



BBC_DF=pd.read_csv(filename, error_bad_lines=False)
# print(BBC_DF.head())
# iterating the columns 
# for col in BBC_DF.columns: 
    # print(col) 
    
# print(BBC_DF["Headline"])

## REMOVE any rows with NaN in them
BBC_DF = BBC_DF.dropna()
# print(BBC_DF["Headline"])

### Tokenize and Vectorize the Headlines
## Create the list of headlines
## Keep the labels!

HeadlineLIST=[]
LabelLIST=[]

for nexthead, nextlabel in zip(BBC_DF["Headline"], BBC_DF["LABEL"]):
    HeadlineLIST.append(nexthead)
    LabelLIST.append(nextlabel)

# print("The headline list is:\n")
# print(HeadlineLIST)

# print("The label list is:\n")
# print(LabelLIST)


##########################################
## Remove all words that match the topics.
## For example, if the topics are food and covid
## remove these exact words.
##
## We will need to do this by hand. 
NewHeadlineLIST=[]

for element in HeadlineLIST:
    # print(element)
    # print(type(element))
    ## make into list
    AllWords=element.split(" ")
    # print(AllWords)
    
    ## Now remove words that are in your topics
    NewWordsList=[]
    for word in AllWords:
        # print(word)
        word=word.lower()
        if word in topics:
            print(word)
        else:
            NewWordsList.append(word)
            
    ##turn back to string
    NewWords=" ".join(NewWordsList)
    ## Place into NewHeadlineLIST
    NewHeadlineLIST.append(NewWords)


##
## Set the     HeadlineLIST to the new one
HeadlineLIST=NewHeadlineLIST
# print(HeadlineLIST)     
#########################################
##
##  Build the labeled dataframe
##
######################################################

### Vectorize
## Instantiate your CV
MyCountV=CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True, 
        stop_words = "english",
        max_features=50
        )

## Use your CV 
MyDTM = MyCountV.fit_transform(HeadlineLIST)  # create a sparse matrix



ColumnNames=MyCountV.get_feature_names()
#print(type(ColumnNames))


## Build the data frame
MyDTM_DF=pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)

## Convert the labels from list to df
Labels_DF = DataFrame(LabelLIST,columns=['LABEL'])


##Save original DF - without the lables
My_Orig_DF=MyDTM_DF

dfs = [Labels_DF, MyDTM_DF]

Final_News_DF_Labeled = pd.concat(dfs,axis=1, join='inner')
   
##############################################
##
##   LDA Topics Modeling
##
##
#########################################################
NumTopics=5
NUM_TOPICS=NumTopics
lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10000, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
   
lda_Z_DF = lda_model.fit_transform(My_Orig_DF)
print(lda_Z_DF.shape)  # (NO_DOCUMENTS, NO_TOPICS)

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                    for i in topic.argsort()[:-top_n - 1:-1]])
 
print("LDA Model:")
print_topics(lda_model, MyCountV)

############## Fancy Plot.................
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pyLDAvis.sklearn as LDAvis
import pyLDAvis

# ##conda install -c conda-forge pyldavis
pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model, MyDTM, MyCountV,  mds='tsne')
pyLDAvis.save_html(panel, "InTheNews.html")

################ Another fun vis for LDA

word_topic = np.array(lda_model.components_)
#print(word_topic)
word_topic = word_topic.transpose()

num_top_words = 15
vocab_array = np.asarray(ColumnNames)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 8

for t in range(NUM_TOPICS):
    plt.subplot(1, NUM_TOPICS, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

plt.tight_layout()
plt.show()
plt.savefig("TopicsVis.pdf")




##################################################################
## Tokenize and Vectorize the text data from the corpus...
##############################################################
## Instantiate three Vectorizers.....
## NOrmal CV
MyVectCount=CountVectorizer(input='content',
                        stop_words='english',
                        max_features=100
                        )

## NOw I can vectorize using my list of complete paths to my files
DTM_Count=MyVectCount.fit_transform(HeadlineLIST)


#####################
## Get the complete vocab - the column names
## !!!!!!!!! FOr TF and CV - but NOT for stemmed...!!!
##################
ColumnNames=MyVectCount.get_feature_names()
print("The vocab is: ", ColumnNames, "\n\n")

## Use pandas to create data frames
DF_Count=pd.DataFrame(DTM_Count.toarray(),columns=ColumnNames)
print(DF_Count)


################################################
##           Let's Cluster........
################################################
# Using sklearn
## you will need
## from sklearn.cluster import KMeans
## import numpy as np
kmeans_object_Count = KMeans(n_clusters=3)
#print(kmeans_object)
kmeans_object_Count.fit(DF_Count)
# Get cluster assignment labels
labels = kmeans_object_Count.labels_
prediction_kmeans = kmeans_object_Count.predict(DF_Count)
#print(labels)
print(prediction_kmeans)
# Format results as a DataFrame
Myresults = pd.DataFrame([DF_Count.index,labels]).T
print(Myresults)

############# ---> ALWAYS USE VIS! ----------
print(DF_Count)

word1 = "launch"
word2 = "discovered"
word3 = "space"
word1_index = DF_Count.columns.get_loc(word1)
word2_index = DF_Count.columns.get_loc(word2)
word3_index = DF_Count.columns.get_loc(word3)
# print(DF_Count["chocolate"]) 
x=DF_Count[word1]  ## col 1  starting from 0
y=DF_Count[word2]    ## col 14  starting from 0
z=DF_Count[word3]  ## col 2  starting from 0
colnames=DF_Count.columns
print(colnames)
#print(x,y,z)


fig1 = plt.figure(figsize=(12, 12))
ax1 = Axes3D(fig1, rect=[0, 0, .90, 1], elev=35, azim=40)

ax1.scatter(x,y,z, cmap="RdYlGn", edgecolor='k', s=200,c=prediction_kmeans)
ax1.w_xaxis.set_ticklabels([])
ax1.w_yaxis.set_ticklabels([])
ax1.w_zaxis.set_ticklabels([])

ax1.set_xlabel(word1, fontsize=25)
ax1.set_ylabel(word2, fontsize=25)
ax1.set_zlabel(word3, fontsize=25)
#plt.show()
        
centers = kmeans_object_Count.cluster_centers_
print(centers)
#print(centers)
C1=centers[0,(word1_index,word2_index,word3_index)]
print(C1)
C2=centers[1,(word1_index,word2_index,word3_index)]
print(C2)
C3=centers[2,(word1_index,word2_index,word3_index)]
print(C3)
xs=C1[0],C2[0],C3[0]
print(xs)
ys=C1[1],C2[1],C3[1]
zs=C1[2],C2[2],C3[2]


ax1.scatter(xs,ys,zs, c='black', s=2000, alpha=0.2)
plt.show()
#plt.cla()






