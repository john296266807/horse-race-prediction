import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_csv('training.csv')

n_win = []
win_rate = []
n_win2 = []
win_rate2 = []
h_id = []
for i in range(0,2155):
    if train.loc[train.horse_index==i].shape[0] != 0: 
        nw = train.loc[train.horse_index==i].loc[train.finishing_position==1].shape[0]
        wr = nw/(train.loc[train.horse_index==i].shape[0])
        n_win.append(nw)
        win_rate.append(wr)
        h_id.append(i)
        
n_win_j = []
win_rate_j = []
j_id = []
for j in range(0,105):
    if train.loc[train.jockey_index==j].shape[0] != 0:
        nwj = train.loc[train.jockey_index==j].loc[train.finishing_position==1].shape[0]
        wrj = nwj/(train.loc[train.jockey_index==j].shape[0])
        n_win_j.append(nwj)
        win_rate_j.append(wrj)
        j_id.append(j)

con_h=[]
for k in range(0,len(n_win)):
    if (win_rate[k]>0.5 and n_win[k]>6):
        con_h.append(1)
    else:
        con_h.append(0)
        
con_j=[]
for l in range(0,len(n_win_j)):
    if (win_rate_j[l]>0.2 and n_win_j[l]>250):
        con_j.append(1)
    else:
        con_j.append(0)

h_c = ['g' if i == 1 else 'b' for i in con_h ]
j_c = ['r' if i == 1 else 'y' for i in con_j ]    


fig, axes = plt.subplots(1,2)
axes[0].scatter(x=win_rate, y=n_win, c=h_c, alpha=0.7)
axes[0].set_title('Horse')
axes[0].set_xlabel('Win rate')
axes[0].set_ylabel('No. of wins')
axes[1].scatter(x=win_rate_j, y=n_win_j, c=j_c, alpha=0.7)
axes[1].set_title('Jockey')
axes[1].set_xlabel('Win rate')
axes[1].set_ylabel('No. of wins')

co = 0
for i in range(0,len(n_win)):
    if con_h[i] == 1:
        axes[0].annotate(train.loc[train.horse_index==h_id[i]]['horse_name'].iloc[0],
                         xy=(win_rate[i],n_win[i]),xytext=(0.5,6.5+co))
        co = co + 1

for i in range(0,len(n_win_j)):
    if con_j[i] == 1:
        axes[1].annotate(train.loc[train.jockey_index==j_id[i]]['jockey'].iloc[0],
                         xy=(win_rate_j[i],n_win_j[i]))

        
plt.suptitle('No. of wins VS Win rate')
fig.tight_layout()
plt.show()