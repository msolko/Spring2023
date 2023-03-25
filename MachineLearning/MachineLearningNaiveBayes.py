# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:30:15 2023

@author: Maxwell Solko
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# COLUMN disc_year:      Discovery Year
# COLUMN rv_flag:        Detected by Radial Velocity Variations
# COLUMN pul_flag:       Detected by Pulsar Timing Variations
# COLUMN ptv_flag:       Detected by Pulsation Timing Variations
# COLUMN tran_flag:      Detected by Transits
# COLUMN ast_flag:       Detected by Astrometric Variations
# COLUMN obm_flag:       Detected by Orbital Brightness Modulations
# COLUMN micro_flag:     Detected by Microlensing
# COLUMN etv_flag:       Detected by Eclipse Timing Variations
# COLUMN ima_flag:       Detected by Imaging
# COLUMN dkin_flag:      Detected by Disk Kinematics
# COLUMN pl_orbper:      Orbital Period [days]
# COLUMN pl_orbsmax:     Orbit Semi-Major Axis [au])
# COLUMN pl_rade:        Planet Radius [Earth Radius]
# COLUMN pl_bmasse:      Planet Mass or Mass*sin(i) [Earth Mass]
# COLUMN pl_orbeccen:    Eccentricity
# COLUMN pl_insol:       Insolation Flux [Earth Flux]
# COLUMN pl_eqt:         Equilibrium Temperature [K]
# COLUMN pl_ratror:      Ratio of Planet to Stellar Radius
# COLUMN st_teff:        Stellar Effective Temperature [K]
# COLUMN sy_dist:        Distance [pc]
data = pd.read_csv('https://raw.githubusercontent.com/msolko/Spring2023/main/MachineLearning/NB%20data.csv')
data.isna().sum()
# After looking through the data, deleted the columns that either had too many NaN or non relevant data
# del data['disc_year']
del data['pl_bmasse']
del data['pl_orbeccen']
del data['pl_insol']
del data['pl_ratror']
# del data['pl_eqt']
data.pl_eqt.describe()
# At first I thought the flags might help, but they are all mostly 0 and dillute the data/accuracy
del data['pul_flag']
del data['ptv_flag']
del data['micro_flag']
del data['etv_flag']
del data['dkin_flag']
del data['tran_flag']
del data['ast_flag']
del data['obm_flag']
del data['rv_flag']
# data.pl_orbeccen.describe()

data.loc[(data['pl_eqt'] < 270) & (data['pl_eqt']>175)]
data = data.dropna().reset_index(drop=True)
# test_df = data.copy()
# test_df = test_df.dropna()
# test_df.loc[(test_df['pl_eqt'] < 270) & (test_df['pl_eqt']>175)]

nb_target = pd.DataFrame()
nb_target['pl_eqt'] = data['pl_eqt'] #have a separate model for the target
del data['pl_eqt'] #get rid of the label from the model
# nb_target

#Change the target to specific labels. In this case, any planet with 
#an equilibrium temperature in a somewhat habitable range is considered
#success, or habitable
checkpoint = nb_target.copy()
nb_target = checkpoint.copy()
habitables_index = nb_target['pl_eqt'].loc[(nb_target['pl_eqt'] < 270) & (nb_target['pl_eqt']>175)].index
habitables_true = nb_target['pl_eqt'][habitables_index]
habitables_true[:] = 'Habitable'
habitables_false = nb_target.drop(habitables_index)
habitables_false[:] = "Inhabitable"

nb_target['pl_eqt'][habitables_true.index] = habitables_true
nb_target['pl_eqt'][habitables_false.index] = habitables_false['pl_eqt']


#######################################
# Naive Bayes
#######################################
X_train, X_test, y_train, y_test = train_test_split(data, nb_target['pl_eqt'], test_size=0.2, random_state=1)

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
plt.title('Confusion Matrix', fontsize=18)
# plt.locator_params(axis='x', nbins=12)
# plt.locator_params(axis='y', nbins=12)
lbls = ['','Habitable','Inhabitable']
ax.xaxis.set_ticklabels(lbls, fontsize = 18)
ax.yaxis.set_ticklabels(lbls, fontsize = 18)
plt.show()




















































