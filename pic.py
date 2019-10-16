import random
import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    path = input("请输入需可视化的文件路径：")
    title = path.split('/')[-1]
    metrics  = [ i for i in input("请分别输入metric1和metric2（，隔开）：").split(',')]
    metric1 = metrics[0]
    metric2 = metrics[1]
    # metric1 = 'dimen1'
    # metric2 = 'dimen2'
    df = pd.read_csv(path,usecols=[metric1,metric2,'label'])
    f = plt.figure(1)

    df1 = df.loc[df['label']==-1]
    df2 = df.loc[df['label']==1]
    plt.scatter(df1[metric1],df1[metric2],color='red',label='anomaly',s=10)
    plt.scatter(df2[metric1],df2[metric2],color='y',label='normal',s=1)
    plt.legend(loc=(1,0))

    # x = df[metric1].tolist()
    # y = df[metric2].tolist()
    # label = df['label'].tolist()
    #
    # label = array(label)+2
    # print(label)
    # plt.scatter(x, y,s = 1 , c=15.0 * label)
    plt.title(title)
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    plt.show()
