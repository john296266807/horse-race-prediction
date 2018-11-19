import pandas as pd
import numpy as np

# import model results
pred = pd.read_csv('reg_pick.csv')

# function for calculating cost and revenue
def cost_rev(pick):
    revm = pd.DataFrame(pred[['finishing_position','win_odds']].copy())
    revm['pick'] = pick.copy()
    rev = sum(revm.loc[revm.pick==1].loc[revm.finishing_position==1]['win_odds'])
    earn = (rev/sum(pick)-1)*100
    print('Cost: ' + str(sum(pick)))
    print('Revenue: ' + str(rev))
    print('Return: ' + str(earn) + '%\n')
 
# betting results of default model results
print('Betting result of SVR (without normalization):')
cost_rev(pred['svr'])
print('Betting result of SVR (with normalization):')
cost_rev(pred['svr_n'])
print('Betting result of GBRT (without normalization):')
cost_rev(pred['gbrt'])
print('Betting result of GBRT (with normalization):')
cost_rev(pred['gbrt_n'])

# new constraint of only betting at win odds larger than 5
new_svr=[]
for i in range(0,pred['svr'].shape[0]):
    if pred['svr'][i]==1 and pred['win_odds'][i]>5:
        new_svr.append(1)
    else:
        new_svr.append(0)

new_gbrt=[]
for i in range(0,pred['gbrt'].shape[0]):
    if pred['gbrt'][i]==1 and pred['win_odds'][i]>5:
        new_gbrt.append(1)
    else:
        new_gbrt.append(0)
        
new_svr_n=[]
for i in range(0,pred['svr_n'].shape[0]):
    if pred['svr_n'][i]==1 and pred['win_odds'][i]>5:
        new_svr_n.append(1)
    else:
        new_svr_n.append(0)
        
new_gbrt_n=[]
for i in range(0,pred['gbrt_n'].shape[0]):
    if pred['gbrt_n'][i]==1 and pred['win_odds'][i]>5:
        new_gbrt_n.append(1)
    else:
        new_gbrt_n.append(0)

print('Betting result of new SVR (without normalization) (with constraint of betting at win odds larger than 5):')
cost_rev(new_svr)
print('Betting result of new SVR (with normalization) (with constraint of betting at win odds larger than 5):')
cost_rev(new_svr_n)
print('Betting result of new GBRT (without normalization) (with constraint of betting at win odds larger than 5):')
cost_rev(new_gbrt)
print('Betting result of new GBRT (with normalization) (with constraint of betting at win odds larger than 5):')
cost_rev(new_gbrt_n)

# import the classification predictions
lr_pred = pd.read_csv('predictions/lr_predictions.csv')
nb_pred = pd.read_csv('predictions/nb_predictions.csv')
rf_pred = pd.read_csv('predictions/rf_predictions.csv')
svm_pred = pd.read_csv('predictions/svm_predictions.csv')

pred['lr'] = lr_pred['HorseRankTop3']
pred['nb'] = nb_pred['HorseWin']
pred['rf'] = rf_pred['HorseWin']
pred['svm'] = svm_pred['HorseRankTop3']

# try to combine different models to pick the winning horse
combine = (0.5*pred['svr']+0.5*pred['svr_n']+0.4*pred['gbrt']+0.4*pred['gbrt_n']+
           0.5*pred['nb']+0.5*pred['rf'])

pred['combine'] = combine.copy()

comb_pick=[]
for i in range(0,5864):
    if pred['combine'][i] == max(pred.loc[pred.race_id==pred['race_id'][i]]['combine']):
        comb_pick.append(1)
    else:
        comb_pick.append(0)

print('Betting result of combining the models:')
cost_rev(comb_pick)

# add constraint of betting at win odds larger than 5 to the new combined model
new_comb_pick=[]
for i in range(0,pred['svr'].shape[0]):
    if comb_pick[i]==1 and pred['win_odds'][i]>5:
        new_comb_pick.append(1)
    else:
        new_comb_pick.append(0)

# add constraint of betting at win odds larger than 10        
print('Betting result of combining the models (with constraint of betting at win odds larger than 5):')
cost_rev(new_comb_pick)

new_comb_pick=[]
for i in range(0,pred['svr'].shape[0]):
    if comb_pick[i]==1 and pred['win_odds'][i]>10:
        new_comb_pick.append(1)
    else:
        new_comb_pick.append(0)

# add constraint of betting at win odds larger than 15  
print('Betting result of combining the models (with constraint of betting at win odds larger than 10):')
cost_rev(new_comb_pick)

new_comb_pick=[]
for i in range(0,pred['svr'].shape[0]):
    if comb_pick[i]==1 and pred['win_odds'][i]>15:
        new_comb_pick.append(1)
    else:
        new_comb_pick.append(0)

print('Betting result of combining the models (with constraint of betting at win odds larger than 15):')
cost_rev(new_comb_pick)
