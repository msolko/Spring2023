# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:13:53 2023

@author: gameb
"""
#Heavily inspired by code from Ami Gates

import requests
import numpy as np
import lxml
import lxml.html
import cssselect
#from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix



# from sklearn import metrics
import sklearn
from sklearn.svm import LinearSVC

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
against_labels = ['Against']*len(against_answers)
for_labels = ['For']*len(for_answers)
# smush the labels into one list
full_labels = for_labels+against_labels

full_dataframe = pd.DataFrame(list(zip(full_labels, full_answers)), columns = ['Label', 'Content'])
debate_points = pd.read_csv('https://raw.githubusercontent.com/msolko/Spring2023/main/TextMining/WesternInstituteSpaceExplorationDebatePoints.csv')


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
# Naive Bayes
#######################################
X_train, X_test, y_train, y_test = train_test_split(FinalDF, label_targets['Label'], test_size=0.2, random_state=2)

myNB = MultinomialNB()
NB1 = myNB.fit(X_train, y_train)
predict1 = myNB.predict(X_test)
cnf_matrix_auth = confusion_matrix(y_test, predict1)

print("Accuracy:",metrics.accuracy_score(y_test, predict1))
print(confusion_matrix(y_test, predict1))
print(classification_report(y_test, predict1))

#make a nicer figure for the confusion matrix
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(cnf_matrix_auth, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cnf_matrix_auth.shape[0]):
    for j in range(cnf_matrix_auth.shape[1]):
        ax.text(x=j, y=i,s=cnf_matrix_auth[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix for Naive Bayes', fontsize=18)
# plt.locator_params(axis='x', nbins=12)
# plt.locator_params(axis='y', nbins=12)
lbls = ['','Against','For']
ax.xaxis.set_ticklabels(lbls, fontsize = 18)
ax.yaxis.set_ticklabels(lbls, fontsize = 18)
plt.show()



#######################################
# Decision Trees
#######################################
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
cnf_matrix_auth = confusion_matrix(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#make a nicer figure for the confusion matrix
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(cnf_matrix_auth, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cnf_matrix_auth.shape[0]):
    for j in range(cnf_matrix_auth.shape[1]):
        ax.text(x=j, y=i,s=cnf_matrix_auth[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix Full Decision Tree', fontsize=18)
# plt.locator_params(axis='x', nbins=12)
# plt.locator_params(axis='y', nbins=12)
lbls = ['','Against','For']
ax.xaxis.set_ticklabels(lbls, fontsize = 18)
ax.yaxis.set_ticklabels(lbls, fontsize = 18)
plt.show()

feature_list = X_train.columns.tolist()
fig=plt.figure(figsize=(40, 25))
tree.plot_tree(clf, max_depth=9, fontsize=16, feature_names=feature_list)

################################# Decision Tree 2, random splits ######################################
clf_random = tree.DecisionTreeClassifier(splitter='random')
clf_random = clf_random.fit(X_train,y_train)
y_pred = clf_random.predict(X_test)
cnf_random_matrix_auth = confusion_matrix(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#make a nicer figure for the confusion matrix
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(cnf_random_matrix_auth, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cnf_random_matrix_auth.shape[0]):
    for j in range(cnf_random_matrix_auth.shape[1]):
        ax.text(x=j, y=i,s=cnf_random_matrix_auth[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix Random Split Tree', fontsize=18)
# plt.locator_params(axis='x', nbins=12)
# plt.locator_params(axis='y', nbins=12)
lbls = ['','Against','For']
ax.xaxis.set_ticklabels(lbls, fontsize = 18)
ax.yaxis.set_ticklabels(lbls, fontsize = 18)
plt.show()

feature_list = X_train.columns.tolist()
fig=plt.figure(figsize=(40, 25))
tree.plot_tree(clf_random, max_depth=9, fontsize=16, feature_names=feature_list)


##################################### Decision Tree 3, set depth ######################################
clf_depth = tree.DecisionTreeClassifier(max_depth=3)
clf_depth = clf_depth.fit(X_train,y_train)
y_pred = clf_depth.predict(X_test)
cnf_depth_matrix_auth = confusion_matrix(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#make a nicer figure for the confusion matrix
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(cnf_depth_matrix_auth, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cnf_depth_matrix_auth.shape[0]):
    for j in range(cnf_depth_matrix_auth.shape[1]):
        ax.text(x=j, y=i,s=cnf_depth_matrix_auth[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix Smaller Depth', fontsize=18)
# plt.locator_params(axis='x', nbins=12)
# plt.locator_params(axis='y', nbins=12)
lbls = ['','Against','For']
ax.xaxis.set_ticklabels(lbls, fontsize = 18)
ax.yaxis.set_ticklabels(lbls, fontsize = 18)
plt.show()

feature_list = X_train.columns.tolist()
fig=plt.figure(figsize=(40, 25))
tree.plot_tree(clf_depth, max_depth=9, fontsize=16, feature_names=feature_list)



#############################################
###########  SVM ############################
#############################################

SVM_Model1=LinearSVC(C=50)
SVM_Model1.fit(X_train, y_train)

print("SVM prediction:\n", SVM_Model1.predict(X_test))
print("Actual:")
print(y_test)

SVM_matrix = confusion_matrix(y_test, SVM_Model1.predict(X_test))
print("\nThe confusion matrix for Linear SVC C=50 is:")
print(SVM_matrix)
print("\n\n")
#make a nicer figure for the confusion matrix
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(SVM_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(SVM_matrix.shape[0]):
    for j in range(SVM_matrix.shape[1]):
        ax.text(x=j, y=i,s=SVM_matrix[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix Linear SVM', fontsize=18)
# plt.locator_params(axis='x', nbins=12)
# plt.locator_params(axis='y', nbins=12)
lbls = ['','Against','For']
ax.xaxis.set_ticklabels(lbls, fontsize = 18)
ax.yaxis.set_ticklabels(lbls, fontsize = 18)
plt.show()

#--------------other kernels
## Sigmoid
SVM_Model2=sklearn.svm.SVC(kernel='sigmoid')
SVM_Model2.fit(X_train, y_train)

print("SVM prediction:\n", SVM_Model2.predict(X_test))
print("Actual:")
print(y_test)

SVM_matrix_sigmoid = confusion_matrix(y_test, SVM_Model2.predict(X_test))
print("\nThe confusion matrix for sigmoid SVM is:")
print(SVM_matrix_sigmoid)
print("\n\n")
#make a nicer figure for the confusion matrix
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(SVM_matrix_sigmoid, cmap=plt.cm.Blues, alpha=0.3)
for i in range(SVM_matrix_sigmoid.shape[0]):
    for j in range(SVM_matrix_sigmoid.shape[1]):
        ax.text(x=j, y=i,s=SVM_matrix_sigmoid[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix Sigmoid SVM', fontsize=18)
# plt.locator_params(axis='x', nbins=12)
# plt.locator_params(axis='y', nbins=12)
lbls = ['','Against','For']
ax.xaxis.set_ticklabels(lbls, fontsize = 18)
ax.yaxis.set_ticklabels(lbls, fontsize = 18)
plt.show()


## POLY
SVM_Model3=sklearn.svm.SVC(C=50, kernel='poly',degree=4)
# print(SVM_Model3)
SVM_Model3.fit(X_train, y_train)

print("SVM prediction:\n", SVM_Model3.predict(X_test))
print("Actual:")
print(y_test)

SVM_matrix_poly = confusion_matrix(y_test, SVM_Model3.predict(X_test))
print("\nThe confusion matrix for SVM poly d=4  is:")
print(SVM_matrix_poly)
print("\n\n")
#make a nicer figure for the confusion matrix
fig, ax = plt.subplots(figsize=(12, 12))
ax.matshow(SVM_matrix_poly, cmap=plt.cm.Blues, alpha=0.3)
for i in range(SVM_matrix_poly.shape[0]):
    for j in range(SVM_matrix_poly.shape[1]):
        ax.text(x=j, y=i,s=SVM_matrix_poly[i, j], va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix Polynomial SVM', fontsize=18)
# plt.locator_params(axis='x', nbins=12)
# plt.locator_params(axis='y', nbins=12)
lbls = ['','Against','For']
ax.xaxis.set_ticklabels(lbls, fontsize = 18)
ax.yaxis.set_ticklabels(lbls, fontsize = 18)
plt.show()

###################################################
##
##   Visualizing the top features
##   Then Visualizing the margin with the top 2 in 2D
##
##########################################################

import matplotlib.pyplot as plt
## Credit: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
## Define a function to visualize the TOP words (variables)
def plot_coefficients(MODEL=SVM_Model1, COLNAMES=X_train.columns, top_features=10):
    ## Model if SVM MUST be SVC, RE: SVM_Model=LinearSVC(C=10)
    coef = MODEL.coef_.ravel()
    top_positive_coefficients = np.argsort(coef,axis=0)[-top_features:]
    top_negative_coefficients = np.argsort(coef,axis=0)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(  x=  np.arange(2 * top_features)  , height=coef[top_coefficients], width=.5,  color=colors)
    feature_names = np.array(COLNAMES)
    plt.xticks(np.arange(0, (2*top_features)), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()
    

plot_coefficients()
plt.savefig('KeyWords.pdf')























