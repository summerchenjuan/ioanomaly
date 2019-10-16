"""
挑选异常数据或者删除label列
"""
import pandas as pd

def anomalydata(dataframe):
    """
    :param dataframe: DataFrame 原始数据
    :return: DataFrame 挑选出的label为-1的数据，即异常数据
    选择异常数据
    """
    return dataframe.loc[dataframe['label']==-1]

def deletelabel(dataframe):
    """
    :param dataframe: DataFrame 原始数据
    :return:DataFrame 删除label列
    删除label列
    """
    return dataframe.drop(['label'],axis=1)

def selectcolumns(dataframe,columns):
    """
    :param dataframe: DataFrame 原始数据
    :param columns:list
    :return:
    """
    return dataframe[columns]

if __name__ == "__main__":
    path = input('请输入文件路径：')
    dataframe = pd.read_csv(path)
    commonlist = ['jobid','starttime','endtime','mdshost','extendhost','offesthost']
    mdslist = commonlist + ['rename', 'setattr', 'getattr', 'statfs', 'mkdir',
               'getxattr', 'sync', 'setxattr', 'mknod', 'link', 'rmdir', 'samedir_rename', 'close', 'unlink',
               'open', 'crossdir_rename']
    extendreadlist = commonlist + ["ext1", "ext2", "ext3", "ext4", "ext5", "ext6",
                  "ext7", "ext8", "ext9", "ext10", "ext11", "ext12", ]
    extendwritelist = commonlist + ["ext13", "ext14", "ext15", "ext16", "ext17",
                  "ext18", "ext19", "ext20", "ext21", "ext22", "ext23", "ext24"]
    offestlist = commonlist + ["off1", "off2"]
    for list in [mdslist,extendreadlist,extendwritelist,offestlist]:
        path2 = input('请输入保存路径：')
        resdata = selectcolumns(dataframe=dataframe,columns=list)
        resdata.to_csv(path2,index=False)
