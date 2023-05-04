# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:54:58 2023

@author: gameb
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import requests
import lxml
import lxml.html
import cssselect
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
# from wordcloud import WordCloud
import nltk

response = requests.get('https://debatewise.org/137-space-exploration-is-a-waste-of-money/')
# print(response.text)


tree = lxml.html.fromstring(response.text)
response_element = tree.xpath('//html//body//div[1]//div//div[1]//main//article//div//div[2]//div[1]//div[2]//p//text()[1]')[0]
response_element = tree.cssselect('title')[0]
# print(response_element.getparent().tag)
body_elem = tree.cssselect('body')[0]
# print(body_elem.getchildren())

test = body_elem.getchildren()[10] #going to div (site grid-container)
# print(test.tag)
test = test.getchildren()[0] #going to div (site-content)
# print(test.tag)
test = test.getchildren()[0] #going to div (content-area)
# print(test.tag)
test = test.getchildren()[0] #going to main (site-main)
# print(test.tag)
test = test.getchildren()[0] #going to article (post-6698)
# print(test.tag)
test = test.getchildren()[0] #going to div (inside-article)
# print(test.tag)
test = test.getchildren()[5] #going to div (entry content)
# print(test.tag)
# print(test.getchildren())
raw_answers = str(test.text_content())


split_by_points = raw_answers.split("\n\n\n\n")

# Yes's in this circumstance are arguing that space is a waste of money. Against Space
# No's are arguing the opposite, space isn't a waste of money. For Space

against_answers = [i for i in split_by_points if("Yes because…" in i)] # only keep the yes answers
for_answers = [i for i in split_by_points if("No because…" in i)] # only keep the no answers

# The Yes and No isn't needed at this point because that list is 'labeled'
against_answers = [answer.split("Yes because…")[1] for answer in against_answers]
against_answers = [x for x in against_answers if x != ''] #get rid of empty answers
for_answers = [answer.split("No because…")[1] for answer in for_answers]
for_answers = [x for x in for_answers if x != ''] #get rid of empty answers
# smush the answers into one list
full_answers = for_answers+against_answers

#create labels for the data
against_labels = [0]*len(against_answers) #Against
for_labels = [1]*len(for_answers) #For
# smush the labels into one list
full_labels = for_labels+against_labels

full_dataframe = pd.DataFrame(list(zip(full_labels, full_answers)), columns = ['Label', 'Content'])
debate_points = pd.read_csv('https://raw.githubusercontent.com/msolko/Spring2023/main/TextMining/WesternInstituteSpaceExplorationDebatePoints.csv')

for_points = debate_points[debate_points['Label'] == "For"]
for_points = [1]*len(for_points)
against_points = debate_points[debate_points['Label'] == "Against"]
against_points = [0]* len(against_points)
debate_points["Label"] = for_points+against_points

data = pd.concat([full_dataframe, debate_points], ignore_index=True,axis=0)
data = data.dropna().reset_index(drop=True)

#separating the label from the data
label_targets = pd.DataFrame() 
label_targets['Label'] = data['Label']
del data["Label"]


MyVectLDA=TfidfVectorizer(input='content')

data_vectorized = MyVectLDA.fit_transform(data['Content'])
ColumnNames=MyVectLDA.get_feature_names()
# print(ColumnNames)
FinalDF=pd.DataFrame(data_vectorized.toarray(),columns=ColumnNames)



nltk.download('stopwords')
#print(nltk.corpus.stopwords.words('english'))
stopword_list = nltk.corpus.stopwords.words('english') 



for nextcol in FinalDF.columns:
    if(re.search(r'[^A-Za-z]+', nextcol)): #get rid of any non alphabetic words
        #print(nextcol)
        FinalDF=FinalDF.drop([nextcol], axis=1)
    elif(len(str(nextcol))<4): #if the word is too short
        #print(nextcol)
        FinalDF=FinalDF.drop([nextcol], axis=1)
    elif(len(str(nextcol))>9): #if the word is too long
        #print(nextcol)
        FinalDF=FinalDF.drop([nextcol], axis=1)
    # elif(nextcol in RemoveWords): #if the word is selected to be cut above
    #     #print(nextcol)
    #     FinalDF=FinalDF.drop([nextcol], axis=1)
    elif(nextcol in stopword_list): #ifthe word is a stopword
        #print(nextcol)
        FinalDF=FinalDF.drop([nextcol], axis=1)



#######################################
# Neural Nets
#######################################
X_train, X_test, y_train, y_test = train_test_split(np.array(FinalDF), label_targets['Label'], test_size=0.2, random_state=2)
X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = tensorflow.keras.models.Sequential()


# print(X_train.shape)
# print(X_train.shape[1:])
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
input_dim = X_train.shape[1]  # Number of features

model = Sequential()
model.add(Dense(8, activation='relu', input_dim=input_dim))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,epochs=50, batch_size=1, verbose=1)


y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test,verbose=1)
print(score)



test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

y_pred[y_pred > .5] = 1
y_pred[y_pred <= .5] = 0


cnf_matrix_auth = confusion_matrix(y_test, y_pred)
#flip it to be in the same form as the other ones from past projects
cnf_matrix_auth = np.flip(cnf_matrix_auth, (0))
cnf_matrix_auth = np.flip(cnf_matrix_auth, (1))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#make a nicer figure for the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(cnf_matrix_auth, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cnf_matrix_auth.shape[0]):
    for j in range(cnf_matrix_auth.shape[1]):
        ax.text(x=j, y=i,s=cnf_matrix_auth[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix for Neural Network', fontsize=18)
# plt.locator_params(axis='x', nbins=12)
# plt.locator_params(axis='y', nbins=12)
lbls = ['','Habitable','Uninhabitable']
ax.xaxis.set_ticklabels(lbls, fontsize = 18)
ax.yaxis.set_ticklabels(lbls, fontsize = 18)
plt.show()

