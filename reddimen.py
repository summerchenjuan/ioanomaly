#数据降维
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def dimenreduction(df,metrics,addmetics):
    """
    :param df: Dataframe 包含多维metrics的dataframe
    :param metrics: list 需要降维的metrics
    :param addmetics:list 需要保留的附加metric信息，例如jobid,startime,endtime,host等
    :return: Dataframe 降维后的dataframe
    """
    data = df[addmetics].values
    ds = df[metrics]
    metricsvalue_list= ds.values
    metricsvalue_list = np.array(metricsvalue_list)
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    metricsvalue_list = scaler.fit_transform(metricsvalue_list)
    print('fit......')

    #t-sne 非线性降维
    tsne = TSNE(n_components=2, verbose=1,init='pca',random_state=40)
    X_embedded = tsne.fit_transform(metricsvalue_list)

    #pca 线性降维
    # pca = PCA(n_components=2, copy=False, random_state=40)
    # X_embedded = pca.fit_transform(metricsvalue_list)

    data = np.concatenate((data, X_embedded), axis=1)
    columns = addmetrics + ['dimen1','dimen2']
    decrease_dimension_df = pd.DataFrame(data,columns=columns)
    return decrease_dimension_df

if __name__ == "__main__":
    path = input("请输入文件路径：")
    pathsave = input("请输入文件保存路径：")
    df = pd.read_csv(path)
    addmetrics = ['jobid','starttime','endtime','mdshost','extendhost','offesthost']
    metrics = ['close','crossdir_rename','getattr','getxattr','link','mkdir','mknod','open','rename','rmdir','samedir_rename','setattr','setxattr','statfs','sync','unlink',
               'ext1','ext10','ext11','ext12','ext13','ext14','ext15','ext16','ext17','ext18','ext19','ext2','ext20','ext21','ext22','ext23','ext24','ext3','ext4','ext5','ext6','ext7','ext8','ext9',
               'off1','off2']
    # metrics = ['ext1','ext10','ext11','ext12','ext13','ext14','ext15','ext16','ext17','ext18','ext19','ext2','ext20','ext21','ext22','ext23','ext24','ext3','ext4','ext5','ext6','ext7','ext8','ext9',
    #            'off1','off2']
    resdf = dimenreduction(df=df,metrics=metrics,addmetics=addmetrics)
    print(resdf)
    resdf.to_csv(pathsave,index=False)
