import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_csv('training.csv')

drawno = []
wr = []
for i in range(1,16):
    drawno.append('Draw '+str(i))
    wr.append(train.loc[train.draw==i].loc[train.finishing_position==1].shape[0])

plt.pie(x=wr,labels=drawno, autopct='%.1f%%')
plt.title('Win rates of different draws')
plt.show()