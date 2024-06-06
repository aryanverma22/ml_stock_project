from pandas import read_csv
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


dataset = read_csv(r'D:\New folder\OneDrive\Documents\Downloads\re-ml.csv')
array = dataset.values

open_price = array[:, 4]

close_price = array[:, 8]

del_percent = array[:, 14]
diff=array[:,15]
turnover=array[:,11]
no_trades=array[:,12]
m=int(len(turnover))

big = max(del_percent)
small=min(del_percent)
ranges= big - small
norm_del_perc = [x /big  for x in del_percent]
# print(norm_del_perc)
all=[diff,turnover,no_trades]
arr=[]


for j in range(m):
    temp = []
    for x in all:
        temp.append(x[j])
    arr.append(temp)

classes = []
for x in norm_del_perc:
    if x <= 0.3:
        classes.append("LOW STOCK DELIVERABLES  ")
    elif x <= 0.66:
        classes.append("AVERAGE STOCK DELIVERABLES  ")
    else:
        classes.append("HIGH STOCK DELIVERABLES")
# zero=one=two=three=four=five=six=seven=0
# for i in del_percent:
#     if i<=20:
#         zero+=1
#     elif i<=30:
#         one+=1
#     elif i<=40:
#         two+=1
#     elif i<=50:
#         three+=1
#     elif i<=60:
#         four+=1
#     elif i<=70:
#         five+=1
#     elif i<=80:
#         six+=1
#     else:
#         seven+=1
# print(zero,one,two,three,four,five,six,seven)





#
x_train, x_validation, y_train, y_validation = train_test_split(arr, classes, test_size=0.2, random_state=1)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

names = []
results = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

model = LinearDiscriminantAnalysis()
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
