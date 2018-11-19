import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.cm as cm
import matplotlib.pyplot as plt

train = pd.read_csv('training.csv')

x_train = train[['actual_weight','declared_horse_weight','draw','win_odds','recent_ave_rank',
                 'jockey_ave_rank','trainer_ave_rank','race_distance']]
win_y_train = np.asarray([train['finishing_position'] == 1 ]).astype(int).reshape(23500,1).ravel()

rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(x_train, win_y_train)

chartdf = pd.DataFrame()
chartdf['label'] = ['Actual\nweight','Declared\nhorse\nweight','Draw','Win\nodds','Horse\naverage\nrank',
                 'Jockey\naverage\nrank','Trainer\naverage\nrank','Race\ndistance']
chartdf['feat_importance'] = rf_model.feature_importances_

chtdf = chartdf.sort_values(by='feat_importance',ascending=False)


plt.bar(x=range(0,8),height=chtdf.feat_importance,tick_label=chtdf.label)
plt.xlabel('Features')
plt.ylabel('Feature importance')
plt.title('Importance of different features')
plt.show()