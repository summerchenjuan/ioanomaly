import pandas as pd
import numpy as np

dataframe = pd.read_csv('D://IHEP/ioanomaly/offest.csv',usecols=['jobid'])
dataframe.drop_duplicates().to_csv('D://IHEP/ioanomaly/afoffest.csv',index=False)

dataframe = pd.read_csv('D://IHEP/ioanomaly/extend.csv',usecols=['jobid'])
dataframe.drop_duplicates().to_csv('D://IHEP/ioanomaly/afextend.csv',index=False)

dataframe = pd.read_csv('D://IHEP/ioanomaly/mds.csv',usecols=['jobid'])
dataframe.drop_duplicates().to_csv('D://IHEP/ioanomaly/afmds.csv',index=False)

dataframe = pd.read_csv('D://IHEP/ioanomaly/afoffest.csv',usecols=['jobid'])
afoffest = dataframe['jobid'].tolist()
dataframe = pd.read_csv('D://IHEP/ioanomaly/afextend.csv',usecols=['jobid'])
afextend = dataframe['jobid'].tolist()
dataframe = pd.read_csv('D://IHEP/ioanomaly/afmds.csv',usecols=['jobid'])
afmds = dataframe['jobid'].tolist()
res = set(afoffest)|(set(afextend))
res = res|(set(afmds))
print(len(set(afoffest)))
print(len(set(afextend)))
print(len(set(afmds)))
print(len(res))