import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVC

train = pd.read_csv('training.csv')

x_train = train[['recent_ave_rank','jockey_ave_rank']]

top50 = []
for i in range(0,23500):
    if (train.iloc[i,0]<=(train[train.race_id==train.iloc[i,18]].shape[0])*0.5):
        top50.append(1)
    else:
        top50.append(0)
        
top50_y_train = np.asarray(top50).astype(int).reshape(23500,1).ravel()

svm_model = SVC(kernel='linear',random_state=0)
svm_model.fit(x_train, top50_y_train)

slope = -svm_model.coef_[0][0]/svm_model.coef_[0][1]
intercept = -svm_model.intercept_[0]/svm_model.coef_[0][1]
b_x = range(0,16)
b_y = b_x*slope+intercept

color=['r' if i == 1 else 'b' for i in top50_y_train]
plt.scatter(x=x_train['recent_ave_rank'],y=x_train['jockey_ave_rank'],c=color,alpha=0.4)
plt.plot(b_x,b_y,label='Classification boundary',c='black')
plt.xlabel('Recent average rank of horse')
plt.ylabel('Recent average rank of jockey')
plt.text(x=0.5,y=1.5,s='The line: Classification boundary\nBlue dots: Not in top 50%\nRed dots: In top 50%')
plt.title('SVM visualization')
plt.show()