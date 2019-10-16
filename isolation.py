import pandas as pd
from sklearn.ensemble import IsolationForest


path = input()
trainpath = input()
trainrespath = input()
prepath = input()
respath = input()
df = pd.read_csv(path)
data = df[['dimen1','dimen2']][:150000].values
model = IsolationForest(n_estimators=100, max_samples=256, contamination=0.01,
                        behaviour='new', verbose=1,
                        n_jobs=1, random_state=40)
model.fit(data)

dataframe = df[:150000]
dataframe.to_csv(trainpath,index=False)
dataframe = pd.read_csv(trainpath)
data = dataframe[['dimen1','dimen2']].values
preres = model.predict(data)
score = model.score_samples(data)*(-1)
dataframe['score'] = score
dataframe['label'] = preres
dataframe.to_csv(trainrespath,index=False)

dataframe = df[150000:]
dataframe.to_csv(prepath,index=False)
dataframe = pd.read_csv(prepath)
data = dataframe[['dimen1','dimen2']].values
preres = model.predict(data)
score = model.score_samples(data)*(-1)
dataframe['score'] = score
dataframe['label'] = preres
dataframe.to_csv(respath,index=False)