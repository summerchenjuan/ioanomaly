import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

def dimenreduction(df,metrics,count):
    """
    :param df: Dataframe 包含多维metrics的dataframe
    :param metrics: list 需要降维的metrics
    :return: Dataframe 降维后的dataframe
    """
    metrics.insert(0, 'label')
    metrics.insert(0, 'endtime')
    metrics.insert(0, 'starttime')
    metrics.insert(0, 'jobid')
    ds = df[metrics]
    lists = ds.values
    metricsvalue_list = lists[:count, 4:]
    label = lists[:count, 3]
    label = np.array(label)
    label = label.reshape(-1)
    metricsvalue_list = np.array(metricsvalue_list)
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    metricsvalue_list = scaler.fit_transform(metricsvalue_list)
    tsne = TSNE(n_components=2, verbose=1,init='pca',random_state=40)
    print('fit......')
    X_embedded = tsne.fit_transform(metricsvalue_list)
    data = lists[:count, :4]
    data = np.concatenate((data, X_embedded), axis=1)
    print(data.shape)
    decrease_dimension_df = pd.DataFrame(data,columns=['jobid','starttime','endtime','label','dimen1','dimen2'])
    # decrease_dimension_df = pd.DataFrame(data, columns=['jobid', 'starttime','endtime','label' 'dimen1', 'dimen2'])
    return decrease_dimension_df

if __name__ == "__main__":
    # path = input("请输入文件路径：")
    path = 'D://IHEP/ioanomaly/1008/predata0.001.csv'
    df = pd.read_csv(path)
    count = int(input("请输入样本数量："))
    choice = int(input("降维所有metric输入1，降维extend输入2："))
    if(choice == 1):
        metrics = ['close','crossdir_rename','getattr','getxattr','link','mkdir','mknod','open','rename','rmdir','samedir_rename','setattr','setxattr','statfs','sync','unlink',
                   'ext1','ext10','ext11','ext12','ext13','ext14','ext15','ext16','ext17','ext18','ext19','ext2','ext20','ext21','ext22','ext23','ext24','ext3','ext4','ext5','ext6','ext7','ext8','ext9',
                   'off1','off2']
    else:
        metrics = ['ext1','ext10','ext11','ext12','ext13','ext14','ext15','ext16','ext17','ext18','ext19','ext2','ext20','ext21','ext22','ext23','ext24','ext3','ext4','ext5','ext6','ext7','ext8','ext9',
                   'off1','off2']
    resdf = dimenreduction(df=df,metrics=metrics,count=count)
    metric1 = 'dimen1'
    metric2 = 'dimen2'
    df = pd.read_csv(path)
    x = resdf[metric1].tolist()
    y = resdf[metric2].tolist()
    label = resdf['label'].tolist()
    f = plt.figure(1)
    label = np.array(label) + 2
    plt.scatter(x, y, s=1, c=15.0 * label)
    title = 'predata'+str(count)+'_'+str(choice)
    plt.title('predata'+title)
    plt.show()
