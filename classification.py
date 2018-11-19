import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from naive_bayes import NaiveBayes 


# import training data
train = pd.read_csv('training.csv')

x_train = train[['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank',
                 'jockey_ave_rank','trainer_ave_rank']]
win_y_train = np.asarray([train['finishing_position'] == 1 ]).astype(int).reshape(23500,1).ravel()
top3_y_train = np.asarray([train['finishing_position'] <= 3 ]).astype(int).reshape(23500,1).ravel()

top50 = []
for i in range(0,23500):
    if (train.iloc[i,0]<=(train[train.race_id==train.iloc[i,18]].shape[0])*0.5):
        top50.append(1)
    else:
        top50.append(0)
        
top50_y_train = np.asarray(top50).astype(int).reshape(23500,1).ravel()

# import testing data
test = pd.read_csv('testing.csv')

x_test = test[['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank',
                 'jockey_ave_rank','trainer_ave_rank']]
win_y_test = np.asarray([test['finishing_position'] == 1 ]).astype(int).reshape(5864,1).ravel()
top3_y_test = np.asarray([test['finishing_position'] <= 3 ]).astype(int).reshape(5864,1).ravel()

top50_2 = []
for i in range(0,5864):
    if (test.iloc[i,0]<=(test[test.race_id==test.iloc[i,18]].shape[0])*0.5):
        top50_2.append(1)
    else:
        top50_2.append(0)
        
top50_y_test = np.asarray(top50_2).astype(int).reshape(5864,1).ravel()

# function for calculating recall and precision
def recall_precision(pred, actual):
    tp = 0
    for i in range(0,len(pred)):
        if pred[i] == 1 and actual[i] == 1:
            tp = tp + 1 
    print("Recall: " + str(tp/sum(actual)))
    if sum(pred) == 0:
        print("Precision: There are no positive predictions\n")
    else:
        print("Precision: " + str(tp/sum(pred))+"\n")

############ Logistic regression ############ 
### function ###
def log_reg(x_train, y_train, x_test, y_test):
    lr_model = LogisticRegression(random_state=0)
    lr_model.fit(x_train, y_train)
    print("The score of Logistic regression (based on testing data): " + 
          str(lr_model.score(x_test, y_test)))
    pred = lr_model.predict(x_test)
    kfold = KFold(n_splits=10, random_state=0)
    print("The score of Cross validation on Logistic regression: \n" +
         str(cross_val_score(lr_model, x_train, y_train,
                             cv=kfold, scoring='accuracy')))
    return pred
    
# on winning
print("Logistic regression result on winning:")
win_y_pred_lr = log_reg(x_train, win_y_train, x_test, win_y_test)
recall_precision(win_y_pred_lr, win_y_test)
# on ranking top 3
print("Logistic regression result on ranking top 3:")
top3_y_pred_lr = log_reg(x_train, top3_y_train, x_test, top3_y_test)
recall_precision(top3_y_pred_lr, top3_y_test)
# on ranking top 50%
print("Logistic regression result on ranking top 50%:")
top50_y_pred_lr = log_reg(x_train, top50_y_train, x_test, top50_y_test)
recall_precision(top50_y_pred_lr, top50_y_test)


############ Naive Bayes ############ 
### function ###
# GaussianNB is chosen
def nb(x_train, y_train, x_test, y_test):
    nb_model = GaussianNB()
    nb_model.fit(x_train, y_train)
    print("The score of Naive bayes (based on testing data): " + 
          str(nb_model.score(x_test, y_test)))
    pred = nb_model.predict(x_test)
    kfold = KFold(n_splits=10, random_state=0)
    print("The score of Cross validation on Naive bayes: \n" +
         str(cross_val_score(nb_model, x_train, y_train,
                             cv=kfold, scoring='accuracy'))+"\n")
    return pred

print("\nGaussian Naive bayes result on winning:")
win_y_pred_nb = nb(x_train, win_y_train, x_test, win_y_test)
recall_precision(win_y_pred_nb, win_y_test)
# on ranking top 3
print("Gaussian Naive bayes result on ranking top 3:")
top3_y_pred_nb = nb(x_train, top3_y_train, x_test, top3_y_test)
recall_precision(top3_y_pred_nb, top3_y_test)
# on ranking top 50%
print("Gaussian Naive bayes result on ranking top 50%:")
top50_y_pred_nb = nb(x_train, top50_y_train, x_test, top50_y_test)
recall_precision(top50_y_pred_nb, top50_y_test)

# My Naive Bayes implementation
def mnb(x_train, y_train, x_test, y_test):
    mynb = NaiveBayes()
    mynb.fit(x_train, y_train)
    pred = mynb.predict(x_test)
    print("The score of my Naive Bayes result (based on testing data): " + 
          str(sum(pred==y_test)/len(pred)))
    return pred

# on winning
print("My Naive Bayes result on winning:")
win_y_pred_mnb = mnb(x_train, win_y_train, x_test, win_y_test)
recall_precision(win_y_pred_mnb, win_y_test)
# on ranking top 3
print("My Naive Bayes result on ranking top 3:")
top3_y_pred_mnb = mnb(x_train, top3_y_train, x_test, top3_y_test)
recall_precision(top3_y_pred_mnb, top3_y_test)
# on ranking top 50%
print("My Naive Bayes result on ranking top 50%:")
top50_y_pred_mnb = mnb(x_train, top50_y_train, x_test, top50_y_test)
recall_precision(top50_y_pred_mnb, top50_y_test)



############ SVM ############ 
### function ###
def supvec(x_train, y_train, x_test, y_test):
    svm_model = SVC(kernel='linear',random_state=0)
    svm_model.fit(x_train, y_train)
    print("The score of SVM (based on testing data): " + 
          str(svm_model.score(x_test, y_test)))
    pred = svm_model.predict(x_test)
    kfold = KFold(n_splits=3, random_state=0)
    print("The score of Cross validation on SVM: \n" +
         str(cross_val_score(svm_model, x_train, y_train,
                             cv=kfold, scoring='accuracy'))+"\n")
    return pred

# on winning
print("SVM result on winning:")
win_y_pred_svm = supvec(x_train, win_y_train, x_test, win_y_test)
recall_precision(win_y_pred_svm, win_y_test)
# on ranking top 3
print("SVM result on ranking top 3:")
top3_y_pred_svm = supvec(x_train, top3_y_train, x_test, top3_y_test)
recall_precision(top3_y_pred_svm, top3_y_test)
# on ranking top 50%
print("SVM result on ranking top 50%:")
top50_y_pred_svm = supvec(x_train, top50_y_train, x_test, top50_y_test)
recall_precision(top50_y_pred_svm, top50_y_test)


############ Random forest ############ 
def rf(x_train, y_train, x_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=1000,random_state=0)
    rf_model.fit(x_train, y_train)
    print("The score of random forest (based on testing data): " + 
          str(rf_model.score(x_test, y_test)))
    pred = rf_model.predict(x_test)
    kfold = KFold(n_splits=5, random_state=0)
    print("The score of Cross validation on random forest: \n" +
         str(cross_val_score(rf_model, x_train, y_train,
                             cv=kfold, scoring='accuracy'))+"\n")
    return pred

# on winning
print("\nRandom forest result on winning:")
win_y_pred_rf = rf(x_train, win_y_train, x_test, win_y_test)
recall_precision(win_y_pred_rf, win_y_test)
# on ranking top 3
print("Random forest result on ranking top 3:")
top3_y_pred_rf = rf(x_train, top3_y_train, x_test, top3_y_test)
recall_precision(top3_y_pred_rf, top3_y_test)
# on ranking top 50%
print("Random forest result on ranking top 50%:")
top50_y_pred_rf = rf(x_train, top50_y_train, x_test, top50_y_test)
recall_precision(top50_y_pred_rf, top50_y_test)


###### write csv files to store predictions ######
idm = test[['race_id','horse_id']]
idm.columns = ["RaceID", "HorseID"]
lr_yhat = idm
lr_yhat['HorseWin'] = win_y_pred_lr
lr_yhat['HorseRankTop3'] = top3_y_pred_lr
lr_yhat['HorseRankTop50Percent'] = top50_y_pred_lr

nb_yhat = idm
nb_yhat['HorseWin'] = win_y_pred_nb
nb_yhat['HorseRankTop3'] = top3_y_pred_nb
nb_yhat['HorseRankTop50Percent'] = top50_y_pred_nb

svm_yhat = idm
svm_yhat['HorseWin'] = win_y_pred_svm
svm_yhat['HorseRankTop3'] = top3_y_pred_svm
svm_yhat['HorseRankTop50Percent'] = top50_y_pred_svm

rf_yhat = idm
rf_yhat['HorseWin'] = win_y_pred_rf
rf_yhat['HorseRankTop3'] = top3_y_pred_rf
rf_yhat['HorseRankTop50Percent'] = top50_y_pred_rf

lr_yhat.to_csv('predictions/lr_predictions.csv', index=False)
nb_yhat.to_csv('predictions/nb_predictions.csv', index=False)
svm_yhat.to_csv('predictions/svm_predictions.csv', index=False)
rf_yhat.to_csv('predictions/rf_predictions.csv', index=False)