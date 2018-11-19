import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import norm
pd.options.mode.chained_assignment = None


class NaiveBayes:
    def fit(self, X, y):
        x1 = X.copy()
        x1['y'] = y
        self.mfor1 = x1[x1['y']==1].mean(0).iloc[0:-1]
        self.mfor0 = x1[x1['y']==0].mean(0).iloc[0:-1]
        self.vfor1 = x1[x1['y']==1].var(0).iloc[0:-1]
        self.vfor0 = x1[x1['y']==0].var(0).iloc[0:-1]
    def predict(self, Xt):
        pred = []
        for j in range(0,Xt.shape[0]):
            p0 = []
            p1 = []
            for i in range(0,Xt.shape[1]):
                p0.append(norm(self.mfor0[i],sqrt(self.vfor0[i])).pdf(Xt.iloc[j][i]))
                p1.append(norm(self.mfor1[i],sqrt(self.vfor1[i])).pdf(Xt.iloc[j][i]))
            if np.prod(p0) >= np.prod(p1):
                pred.append(0)
            else:
                pred.append(1)
        return pred
              