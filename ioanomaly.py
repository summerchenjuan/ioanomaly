import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

def buildmodel(n_estimators=100, max_samples=256, contamination=0.001):
    """
    :param n_estimators: int default=100 树的个数
    :param max_samples: int default=256 子采样的个数
    :param contamination: float default=0.01 异常百分比
    :return: model 构造的iforest模型
    """
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, behaviour='new', verbose=1,
                            n_jobs=1, random_state=40)
    return model

def modelfit(model,dataframe,flist):
    """
    :param model: IsolationForest
    :param dataframe: DataFrame 训练样本
    :param flist:list 特征
    :return: IsolationForest 训练好的模型
    训练模型
    """
    data = dataframe[flist].values
    model.fit(data)
    return model

def modelpredict(model,dataframe,flist):
    if(not dataframe.empty):
        data = dataframe[flist].values
        preres = model.predict(data)
        score = model.score_samples(data)*-1
        dataframe['score'] = score
        dataframe['label'] = preres
    else:
        dataframe['score'] = None
        dataframe['label'] = None
    return dataframe

def save(model,path):
    """
    :param model: 模型
    :param path:str 模型保存的路径 文件类型为.sav
    利用pickle保存模型
    """
    pickle.dump(model, open(path, 'wb'))

def load(path):
    """
    :param path: 模型路径
    :return: model
    """
    model = pickle.load(open(path, 'rb'))
    return model

if __name__ == "__main__":
    commonlist = ['jobid', 'starttime', 'endtime', 'mdshost', 'extendhost', 'offesthost']
    # path = input("输入文件路径:")
    # trainsample = int(input("请输入训练样本数目:"))
    # trainrespath = input("输入训练集结果文件路径:")
    # prerespath = input("输入测试集结果文件路径:")
    path_ = 'traindata.csv'
    trainsample = 438844
    trainrespath_ = 'trainlabeldata.csv'
    prerespath_ = 'prelabeldata.csv'
    for file in ['mds','extendread','extendwrite','offest']:
        path = 'D://IHEP/ioanomaly/1005-1010/fourtraindata/'+file+'/'+path_
        trainrespath = 'D://IHEP/ioanomaly/1005-1010/fourtraindata/' + file + '/438844/' + trainrespath_
        prerespath = 'D://IHEP/ioanomaly/1005-1010/fourtraindata/' + file + '/438844/' + prerespath_
        df = pd.read_csv(path)
        flist = df.columns.tolist()
        dataframe = df[:trainsample]
        #flist = ['dimen1', 'dimen2']
        condition = lambda x:x not in commonlist
        flist = list(filter(condition,flist))
        print(flist)
        model = buildmodel()
        model = modelfit(model=model,dataframe=dataframe,flist=flist)

        dataframe = df[:trainsample]
        resframe1 = modelpredict(model=model,dataframe=dataframe,flist=flist)
        resframe1.to_csv(trainrespath, index=False)

        dataframe = df[trainsample:]
        resframe2 = modelpredict(model=model,dataframe=dataframe,flist=flist)
        resframe2.to_csv(prerespath, index=False)





