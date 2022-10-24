# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 12:41:52 2022

@author: Mohammad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

data= pd.read_csv(r"C:\Users\Mohammad\Downloads\train.csv")
data=data.drop(["Genus" , "Species", "RecordID","MFCCs_ 1"], axis =1)
x_data=data.drop(["Family"],axis=1)
y_data=data["Family"]
y_data[y_data == "Leptodactylidae"] = 0
y_data[y_data == "Hylidae"] = 1
y_data[y_data == "Dendrobatidae"] = 2
y_data[y_data == "Bufonidae"] = 3
x_data=x_data.to_numpy()
y_data=y_data.to_numpy()
x_data=np.array(x_data, dtype=float)
y_data=np.array(y_data,dtype=float)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3,random_state=3)


#A

start_time = time.time()
gnb = LinearSVC()
gnb=SVC(kernel='linear')
gnb.fit(x_train, y_train)
clock = time.time() - start_time
y_train_pred = gnb.predict(x_train)
acc_train = 100 * accuracy_score(y_train, y_train_pred)

y_test_pred = gnb.predict(x_test)
acc_test = 100 * accuracy_score(y_test, y_test_pred)
print('Training Time :', str(round(clock, 2)), 'Seconds')
print('Train Accuracy is:', str(round(acc_train, 2)), '%')
print('Test Accuracy is:',str(round(acc_test, 2)), '%')
print("Number of support vectors:",str(len(gnb.support_vectors_[:, :])))
plt.plot(acc_train,'r')
plt.plot(acc_test,'c')
plt.ylabel('Accuracy (%)')
plt.xlabel('Iteration')
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
plt.grid()
plt.title('Model Accuracy')
plt.show()




#B

C = np.asarray([ 0.001,0.01, 0.1, 1,5, 100])
x_train_v, x_val, y_train_v, y_val = train_test_split(x_train, y_train, test_size=0.3)
kf = KFold(n_splits=4, shuffle=True)
acc_train_c_list = []
acc_val_c_list = []
for c in C:
    acc_train_fold_list = []
    acc_val_fold_list = []
    for train_index, valid_index in kf.split(x_train):
        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_val_fold = x_train[valid_index]
        y_val_fold = y_train[valid_index]
        gnb = SVC(C=c, kernel='linear')
        gnb.fit(x_train_fold, y_train_fold.ravel())
        y_train_fold_pred = gnb.predict(x_train_fold)
        y_val_fold_pred = gnb.predict(x_val_fold)
        acc_train_fold = 100 * accuracy_score(y_train_fold, y_train_fold_pred)
        acc_val_fold = 100 * accuracy_score(y_val_fold, y_val_fold_pred)
        acc_train_fold_list.append(acc_train_fold)
        acc_val_fold_list.append(acc_val_fold)
    acc_train_c_list.append(sum(acc_train_fold_list) / len(acc_train_fold_list))
    acc_val_c_list.append(sum(acc_val_fold_list) / len(acc_val_fold_list))

acc_val_max = max(acc_val_c_list)
acc_val_max_index = acc_val_c_list. index(acc_val_max)
acc_train_max = acc_train_c_list[acc_val_max_index]
acc_val_max_c = C[acc_val_max_index]

print('---------------------')
print('C with Maximum Validation Accuracy :', acc_val_max_c)
print('Maximum Training Accuracy :', str(round(acc_train_max, 2)), '%')
print('Maximum Validation Accuracy :', str(round(acc_val_max, 2)), '%')
print('---------------------')

fig, ax = plt.subplots()
ax.plot(np.asarray(acc_val_c_list), label='Validation Accuracy')
ax.set_xticklabels(('-', '0.001', '0.01', '0.1', '1', '5', '100'))
ax.legend(loc='best')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.grid()

gnb = SVC(C=100, kernel='linear')
gnb.fit(x_train, y_train)
y_train_perd = gnb.predict(x_train)
y_test_perd = gnb.predict(x_test)
acc_train = 100 * accuracy_score(y_train, y_train_perd)
acc_train = 100 * accuracy_score(y_test, y_test_perd)

# Print Result

print('Train Accuracy :', str(round(acc_train, 2)), '%')
print('Number of Support Vectors :',len(gnb.support_vectors_))



#c


x_train_v, x_val, y_train_v, y_val = train_test_split(x_train, y_train, test_size=0.3)

Poly2_SVM = SVC(kernel='poly', degree=2)
Poly2_SVM.fit(x_train_v, y_train_v.ravel())
Poly3_SVM = SVC(kernel='poly', degree=3)
Poly3_SVM.fit(x_train_v, y_train_v.ravel())
Poly4_SVM = SVC(kernel='poly', degree=4)
Poly4_SVM.fit(x_train_v, y_train_v.ravel())
Poly5_SVM = SVC(kernel='poly', degree=5)
Poly5_SVM.fit(x_train_v, y_train_v.ravel())
RBF_SVM = SVC(kernel='rbf')
RBF_SVM.fit(x_train_v, y_train_v.ravel())

# For Poly2
y_val_poly2_pred = Poly2_SVM.predict(x_val)
acc_val_poly2 = 100 * accuracy_score(y_val, y_val_poly2_pred)
# For Poly3
y_val_poly3_pred = Poly3_SVM.predict(x_val)
acc_val_poly3 = 100 * accuracy_score(y_val, y_val_poly3_pred)
# For Poly4
y_val_poly4_pred = Poly4_SVM.predict(x_val)
acc_val_poly4 = 100 * accuracy_score(y_val, y_val_poly4_pred)
# For Poly5
y_val_poly5_pred = Poly5_SVM.predict(x_val)
acc_val_poly5 = 100 * accuracy_score(y_val, y_val_poly5_pred)
# For RBF
y_val_rbf_pred = RBF_SVM.predict(x_val)
acc_val_rbf = 100 * accuracy_score(y_val, y_val_rbf_pred)

print('---------------------')
print('Validation Accuracy for Poly2 Kernel :', str(round(acc_val_poly2, 2)), '%')
print('Number of Support Vectors for Poly2 Kernel :',len(Poly2_SVM.support_vectors_))
print('---------------------')
print('Validation Accuracy for Poly3 Kernel :', str(round(acc_val_poly3, 2)), '%')
print('Number of Support Vectors for Poly3 Kernel :',len(Poly3_SVM.support_vectors_))
print('---------------------')
print('Validation Accuracy for Poly4 Kernel :', str(round(acc_val_poly4, 2)), '%')
print('Number of Support Vectors for Poly4 Kernel :',len(Poly4_SVM.support_vectors_))
print('---------------------')
print('Validation Accuracy for Poly5 Kernel :', str(round(acc_val_poly5, 2)), '%')
print('Number of Support Vectors for Poly5 Kernel :',len(Poly5_SVM.support_vectors_))
print('---------------------')
print('Validation Accuracy for RBF Kernel :', str(round(acc_val_rbf, 2)), '%')
print('Number of Support Vectors for RBF Kernel :',len(RBF_SVM.support_vectors_))
print('---------------------')



#D

C = np.asarray([ 0.001,0.01, 0.1, 1,5, 100])

x_train_v, x_val, y_train_v, y_val = train_test_split(x_train, y_train, test_size=0.3)

acc_val_list = []
for c in C:
  RBF_SVM = SVC(C=c, kernel='rbf')
  RBF_SVM.fit(x_train_v, y_train_v)
  y_val_perd = RBF_SVM.predict(x_val)
  acc_val = 100 * accuracy_score(y_val, y_val_perd)
  acc_val_list.append(acc_val)

acc_val_max = max(acc_val_list)
acc_val_max_index = acc_val_list. index(acc_val_max)
acc_val_max_c = C[acc_val_max_index]

print('---------------------')
print('C with Maximum Validation Accuracy :', acc_val_max_c)
print('Maximum Validation Accuracy :', str(round(acc_val_max, 2)), '%')
print('---------------------')

fig, ax = plt.subplots()
ax.plot(np.asarray(acc_val_list), label='Validation Accuracy')
ax.set_xticklabels(('-', '0.001', '0.01', '0.1', '1', '5', '100'))
ax.legend(loc='lower right')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy of RBF SVM')
plt.grid()

RBF_SVM = SVC(C=0.001, kernel='rbf')
RBF_SVM.fit(x_train, y_train.ravel())
y_train_perd = RBF_SVM.predict(x_train)
y_test_perd = RBF_SVM.predict(x_test)
acc_train = 100 * accuracy_score(y_train, y_train_perd)
acc_test = 100 * accuracy_score(y_test, y_test_perd)

# Print Result
print('---------------------')
print('Train Accuracy :', str(round(acc_train, 2)), '%')
print('Test Accuracy :', str(round(acc_test, 2)), '%')
print('Number of Support Vectors :',len(RBF_SVM.support_vectors_))
# print('---------------------')





#E

C = np.asarray([ 0.001,0.01, 0.1, 1,5, 100])
kf = KFold(n_splits=4, shuffle=True)
acc_train_c_list = []
acc_val_c_list = []
for c in C:
    acc_train_fold_list = []
    acc_val_fold_list = []
    for train_index, valid_index in kf.split(x_train):
        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_val_fold = x_train[valid_index]
        y_val_fold = y_train[valid_index]
        gnb = SVC(C=c, kernel='rbf')
        gnb.fit(x_train_fold, y_train_fold.ravel())
        y_train_fold_pred = gnb.predict(x_train_fold)
        y_val_fold_pred = gnb.predict(x_val_fold)
        acc_train_fold = 100 * accuracy_score(y_train_fold, y_train_fold_pred)
        acc_val_fold = 100 * accuracy_score(y_val_fold, y_val_fold_pred)
        acc_train_fold_list.append(acc_train_fold)
        acc_val_fold_list.append(acc_val_fold)
    acc_train_c_list.append(sum(acc_train_fold_list) / len(acc_train_fold_list))
    acc_val_c_list.append(sum(acc_val_fold_list) / len(acc_val_fold_list))

acc_val_max = max(acc_val_c_list)
acc_val_max_index = acc_val_c_list. index(acc_val_max)
acc_train_max = acc_train_c_list[acc_val_max_index]
acc_val_max_c = C[acc_val_max_index]

print('---------------------')
print('C with Maximum Validation Accuracy :', acc_val_max_c)
print('Maximum Validation Accuracy :', str(round(acc_val_max, 2)), '%')
print('---------------------')

fig, ax = plt.subplots()
ax.plot(np.asarray(acc_val_c_list), label='Validation Accuracy')
ax.set_xticklabels(('-', '0.001', '0.01', '0.1', '1', '5', '100'))
ax.legend(loc='lower right')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.grid()