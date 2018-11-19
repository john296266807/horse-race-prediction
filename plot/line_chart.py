import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

train = pd.read_csv('training.csv')

# plot recent six races of a horse
print("Please enter the horse id to know its recent six games: ")
hid = input()

ifg = train.loc[train.horse_id==hid].shape[0]
if ifg >= 6:
    r6r_y = train.loc[train.horse_id==hid].iloc[-6:].finishing_position.copy()
    r6r_x = train.loc[train.horse_id==hid].iloc[-6:].race_id.copy()
else:
    r6r_y = train.loc[train.horse_id==hid].iloc[-ifg:].finishing_position.copy()
    r6r_x = train.loc[train.horse_id==hid].iloc[-ifg:].race_id.copy()

plt.plot(r6r_x, r6r_y, '-bo')
plt.xlabel('Race ID')
plt.ylabel('Finishing position')
plt.title('Recent race result of ' + hid)
plt.show()