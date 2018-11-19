import pandas as pd
import numpy as np

data = pd.read_csv("race-result-horse.csv")
# remove samples with no rank
data = data[pd.to_numeric(data['finishing_position'], errors='coerce').notnull()]

# add new column showing the positions in recent six runs
r6r =[]
for i in range(0,29364):
    rsr = ''
    data2 = data.iloc[0:i]
    rsrm = data2.loc[data2.horse_id == data.iloc[i, 3]]

    if rsrm.shape[0] >= 6:
        rk = 6
    else:
        rk = rsrm.shape[0]
    
    for j in range(1,rk+1):
        if j != 1:
            rsr = '/' + rsr 
        rsr = rsrm.iloc[-j,0] + rsr
    r6r.append(rsr)

data['recent_6_runs'] = np.asarray(r6r)

# add new column showing average of recent six ranks
avgr6r = []
for k in range(0,29364):
    splitr6r = data.iloc[k,19].split("/")
    splitr6r_flt = [float(n) for n in splitr6r if n]
    avgr6r.append(sum(splitr6r_flt)/len(splitr6r_flt) if splitr6r_flt else 7)

data['recent_ave_rank'] = np.asarray(avgr6r)

# give an index to each horse
horse_index = dict.fromkeys(set(data.iloc[:,3]))
hindex = 0
for horse_id in set(data.iloc[:,3]):
    horse_index[horse_id] = hindex
    hindex = hindex + 1
h_i = []
for i in data['horse_id']:
    h_i.append(horse_index[i])
data['horse_index'] = h_i

# give an index to each jockey
jockey_index = dict.fromkeys(set(data.iloc[:,4]))
jindex = 0
for j in set(data.iloc[:,4]):
    jockey_index[j] = jindex
    jindex = jindex + 1
j_i = []
for i in data['jockey']:
    j_i.append(jockey_index[i])
data['jockey_index'] = j_i
    
# give an index to each trainer
trainer_index = dict.fromkeys(set(data.iloc[:,5]))
tindex = 0
for j in set(data.iloc[:,5]):
    trainer_index[j] = tindex
    tindex = tindex + 1
t_i = []
for i in data['trainer']:
    t_i.append(trainer_index[i])
data['trainer_index'] = t_i

# add a column showing average rank of jockey
df2 = data.iloc[0:23500]
j_averank =[]
for a in range(0,29364):
    j_m = [int(b) for b in df2[df2.jockey==data.iloc[a,4]]['finishing_position']]
    j_mean = np.mean(j_m) if len(j_m) else 7
    j_averank.append(j_mean)
data['jockey_ave_rank'] = j_averank


# add a column showing average rank of trainer
t_averank =[]
for a in range(0,29364):
    t_m = [int(b) for b in df2[df2.trainer==data.iloc[a,5]]['finishing_position']]
    t_mean = np.mean(t_m) if len(t_m) else 7
    t_averank.append(t_mean)
data['trainer_ave_rank'] = t_averank

# add a column showing race distance
distance_data = pd.read_csv("race-result-race.csv")
race_d =[]
for i in range(0,29364):
    race_d.append(int(distance_data.loc[distance_data['race_id']==data.iloc[i,18]]['race_distance']))
data['race_distance'] = race_d

# write training and testing csv files
data.iloc[0:23500].to_csv('training.csv', index = False)
data.iloc[0:23500].to_csv('plot/training.csv', index = False)
data.iloc[23500:].to_csv('testing.csv', index = False)