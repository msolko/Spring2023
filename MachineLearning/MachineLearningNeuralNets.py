# -*- coding: utf-8 -*-
"""


@author: Maxwell Solko
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

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
habitables_true[:] = 1 #Habitable
habitables_false = nb_target.drop(habitables_index)
habitables_false[:] = 0 #Uninhabitable

nb_target['pl_eqt'][habitables_true.index] = habitables_true
nb_target['pl_eqt'][habitables_false.index] = habitables_false['pl_eqt']



# Scaled values for the neural nets. The column discovery year won't make sense, but I'm keeping it in for now.
x = data.values #returns a numpy array
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
final_df = pd.DataFrame(x_scaled, columns=data.columns)


#######################################
# Neural Nets
#######################################
X_train, X_test, y_train, y_test = train_test_split(data, nb_target['pl_eqt'], test_size=0.2, random_state=1)

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = tensorflow.keras.models.Sequential()
# print(X_train.shape)
# print(X_train.shape[1:])
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(7,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
history = model.fit(X_train, y_train,epochs=8, batch_size=1, verbose=1)


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

