import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from math import sqrt
from heapq import nlargest
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# import data
train = pd.read_csv('training.csv')
test = pd.read_csv('testing.csv')

x_train = train[['actual_weight','declared_horse_weight','draw','win_odds',
                 'jockey_ave_rank','trainer_ave_rank','recent_ave_rank','race_distance']]
x_test = test[['actual_weight','declared_horse_weight','draw','win_odds',
                 'jockey_ave_rank','trainer_ave_rank','recent_ave_rank','race_distance']]

# import finishing time and change format from m.s.ms to second
y_train = []
for i in train['finish_time'].str.split('.'):
    y_train.append(int(i[0])*60+int(i[1])+int(i[2])/100)
    
y_test = []
for i in test['finish_time'].str.split('.'):
    y_test.append(int(i[0])*60+int(i[1])+int(i[2])/100)

# Function for calculating rmse, top_1, top_3 and average_rank
def evaluate(name, pred):
    rmse = sqrt(sum(np.square(pred-y_test))/len(pred))
    pmatrix = pd.DataFrame(test['race_id'].copy())
    pmatrix['position'] = test['finishing_position'].copy()
    pmatrix['pred_time']=pred.copy()
    p_top1=[]
    for i in range(0,pmatrix.shape[0]):
        if pmatrix.iloc[i,2]==min(pmatrix.loc[pmatrix.race_id==pmatrix.iloc[i,0]].iloc[:,2]):
            p_top1.append(1)
        else:
            p_top1.append(0)

    pmatrix['ptop1']=p_top1
    top1 = sum(pmatrix['position']==p_top1)/sum(p_top1)
    top3 = sum(pmatrix[pmatrix['ptop1']==1]['position'] <= 3)/sum(p_top1)
    ar = np.mean(pmatrix.loc[pmatrix.ptop1==1]['position'])
    return name, rmse, top1, top3, ar
    

# SVR
svr_model = SVR(kernel='linear', C=0.1)
svr_model.fit(x_train, y_train)
print("The SVR score on testing data: " + str(svr_model.score(x_test, y_test)))
kfold = KFold(n_splits=3,random_state=0)
print("The score of Cross validation on SVR: \n" +
         str(cross_val_score(svr_model, x_train, y_train,
                             cv=kfold, scoring='neg_mean_squared_error'))+"\n")
y_svr = svr_model.predict(x_test)
print ("SVR (not normalized) results: " +
       str(evaluate('svr_model', y_svr)))

# GBRT
gbrt_model = GradientBoostingRegressor(loss='quantile',random_state=0,learning_rate=0.2,n_estimators=300, max_depth=2)
gbrt_model.fit(x_train, y_train)
print("The GBRT score on testing data: " + str(gbrt_model.score(x_test, y_test)))
kfold = KFold(n_splits=5,random_state=0)
print("The score of Cross validation on SVR: \n" +
         str(cross_val_score(gbrt_model, x_train, y_train,
                             cv=kfold, scoring='neg_mean_squared_error'))+"\n")
y_gbrt = gbrt_model.predict(x_test)
print ("GBRT (not normalized) results: " +
       str(evaluate('gbrt_model', y_gbrt)))


# Standardize data
xscale = StandardScaler()
x_train_scale = xscale.fit_transform(x_train) 
x_test_scale = xscale.transform(x_test)

yscale = StandardScaler()
y_train_scale = yscale.fit_transform(np.asarray(y_train).reshape(-1, 1))
y_test_scale = yscale.transform(np.asarray(y_test).reshape(-1, 1))


# SVR normalized
svr_model_n = SVR(kernel='linear', C=0.1)
svr_model_n.fit(x_train_scale, y_train_scale.ravel())
print("The SVR score on testing data: " + str(svr_model_n.score(x_test_scale, y_test_scale.ravel())))
kfold = KFold(n_splits=3,random_state=0)
print("The score of Cross validation on SVR (with normalisation): \n" +
         str(cross_val_score(svr_model_n, x_train_scale, y_train_scale.ravel(),
                             cv=kfold, scoring='neg_mean_squared_error'))+"\n")
y_svr_n = yscale.inverse_transform(svr_model_n.predict(x_test_scale))
print ("SVR (normalized) results: " +
       str(evaluate('svr_model_normalized', y_svr_n)))

# GBRT normalized
gbrt_model_n = GradientBoostingRegressor(loss='quantile',random_state=0,learning_rate=0.2,n_estimators=200, max_depth=2)
gbrt_model_n.fit(x_train_scale, y_train_scale.ravel())
print("The GBRT score on testing data: " + str(gbrt_model_n.score(x_test_scale, y_test_scale.ravel())))
kfold = KFold(n_splits=5,random_state=0)
print("The score of Cross validation on GBRT (with normalisation): \n" +
         str(cross_val_score(gbrt_model_n, x_train_scale, y_train_scale.ravel(),
                             cv=kfold, scoring='neg_mean_squared_error'))+"\n")
y_gbrt_n = yscale.inverse_transform(gbrt_model_n.predict(x_test_scale))
print ("GBRT (normalized) results: " +
       str(evaluate('gbrt_model_normalized', y_gbrt_n)))

# export predictions to reg_pick.csv
def pickwin(time):
    pm = pd.DataFrame(test['race_id'].copy())
    pm['pred_time']=time.copy()
    pick=[]
    for i in range(0,pm.shape[0]):
        if pm.iloc[i,1]==min(pm.loc[pm.race_id==pm.iloc[i,0]].iloc[:,1]):
            pick.append(1)
        else:
            pick.append(0)
    return pick

ptimem = test[['race_id','horse_id','finishing_position','win_odds']].copy()

ptimem['svr'] = pickwin(y_svr.copy())
ptimem['gbrt'] = pickwin(y_gbrt.copy())
ptimem['svr_n'] = pickwin(y_svr_n.copy())
ptimem['gbrt_n'] = pickwin(y_gbrt_n.copy())

ptimem.to_csv('reg_pick.csv', index=False)