from elasticsearch import Elasticsearch
from elasticsearch.exceptions import TransportError
from elasticsearch.helpers import bulk, streaming_bulk
import pandas as pd
import time,datetime
from sklearn.ensemble import IsolationForest
import pickle

def search_bulk(index,query_json):
    """
    :param index: str es中的index
    :param query_json: 查询语句
    :return: es中查询到的相应数据
    """
    queryData = es.search(index=index, scroll='5m', timeout='3s', size=1000, body=query_json)
    mdata = queryData.get("hits").get("hits")
    if not mdata:
        #该时间段的数据为空
        columns = query_json['_source']['includes']
        columns.remove('host')
        # print(query_json)
        mdata = pd.DataFrame(columns=columns)
    scroll_id = queryData["_scroll_id"]
    total = queryData["hits"]["total"]
    for i in range(int(total / 100)):
        res = es.scroll(scroll_id=scroll_id, scroll='5m')  # scroll参数必须指定否则会报错
        mdata += res["hits"]["hits"]
    # 打印获取的es数据
    return mdata

def mdata_dataframe(mdata):
    '''
    将从es查询获取的数据解析为DataFrame形式
    :param mdata:从es查询获取的数据
    :return:DataFrame类型
    '''
    if(type(mdata) == list):
        sourcediclist = [dic['_source'] for dic in mdata]
        source = pd.DataFrame(sourcediclist)
        source.rename(columns={'@timestamp': 'timestamp'}, inplace=True)
        source['jobid'] = source['jobid'].fillna(-1)
        source = source.drop(source[source.jobid==-1].index.tolist())
        # source.fillna(0)
        source['timestamp'] = source['timestamp'].map(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ") + datetime.timedelta(hours=8))
        source.drop_duplicates(inplace=True)
        return source
    else:
        mdata.rename(columns={'@timestamp': 'timestamp'}, inplace=True)
        mdata['timestamp'] = mdata['timestamp'].astype(datetime.datetime)
        return mdata


def offest_queryjson(starttime, endtime):
    '''
    :param starttime: datetime timestamp
    :param endtime:datetime timestamp
    :return:query_json
    客户端的/tmp/*fs_offset (注意每个客户端有多个*fs_offset)
    格式为：
    ts:1569466201,fs_name:acfs,job_id:44637629.0,0,101
    三个字符串，加2个整数
    '''
    offsetfields = {"includes": ["host", "@timestamp", "host", "fsname", "jobid", "off1", "off2"], "excludes": []}
    query_json = {
        "_source": offsetfields,
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": '*',
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": starttime,
                                "lte": endtime,
                                "format": "epoch_millis"
                            }
                        },
                    },
                ],
                "filter": [],
                "should": [],
                "must_not": [],
            }
        },
        "sort": {
            "@timestamp": {
                "order": "asc"
            }
        }
    }
    return query_json

def extend_queryjson(starttime,endtime):
    """
    客户端的/tmp/*fs_extent (注意每个客户端有多个*fs_extent)
    格式为：
    ts:1569465901,fs:bprofs,job_id:45961737.0,8,1,0,1,0,0,2,2,3,6,7,0,20941,0,0,0,0,0,0,0,0,0,0,0
    三个字符串，加24个整数
    """
    extentfields = {
        "includes": ["host", "@timestamp", "host", "fsname", "jobid", "ext1", "ext2", "ext3", "ext4", "ext5", "ext6",
                     "ext7", "ext8", "ext9", "ext10", "ext11", "ext12", "ext13", "ext14", "ext15", "ext16", "ext17",
                     "ext18", "ext19", "ext20", "ext21", "ext22", "ext23", "ext24"], "excludes": []}
    query_json = {
        "_source": extentfields,
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": '*',
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": starttime,
                                "lte": endtime,
                                "format": "epoch_millis"
                            }
                        },
                    },
                ],
                "filter": [],
                "should": [],
                "must_not": [],
            }
        },
        "sort": {
            "@timestamp": {
                "order": "asc"
            }
        }
    }
    return query_json


def mds_queryjson(starttime,endtime):
    """
    mds的/tmp/$fsname_elk (为了方便区分，加了一个_elk后缀)
    格式为：
    ts:1569465562,,fsname:bprofs,job_id:45968442.00.005348,0.000000,0.005348,0.000000,0.000000,0.000000,0.000000,0.005348,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000
    三个字符串，加16个浮点数
    """
    mdsfields = {
        "includes": ["host", "@timestamp", "host", "fsname", "jobid", "rename", "setattr", "getattr", "statfs", "mkdir",
                     "getxattr", "sync", "setxattr", "mknod", "link", "rmdir", "samedir_rename", "close", "unlink",
                     "open", "crossdir_rename"], "excludes": []}
    query_json = {
        "_source": mdsfields,
        "query": {
            "bool": {
                "must": [
                    {
                        "query_string": {
                            "query": '*',
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": starttime,
                                "lte": endtime,
                                "format": "epoch_millis"
                            }
                        },
                    },
                ],
                "filter": [],
                "should": [],
                "must_not": [],
            }
        },
        "sort": {
            "@timestamp": {
                "order": "asc"
            }
        }
    }
    return query_json

def createdata(mdsdataframe,extenddataframe,offestdataframe,starttime,endtime):
    """
    :param dataframe1: DataFrame mds
    :param dataframe2: DataFrame extend
    :param dataframe3: DataFrame offest
    :param starttime: int timestamp
    :param endtime: int timestamp
    :return: DataFrame 某段时间序列里的样本
    同一时间段内，将mds extend offest中相同jobid的特征值累加，再对相同的jobid的三个来源的特征值进行拼接
    """

    starttime = starttime/1000
    time_struct = time.localtime(starttime)
    starttime = time.strftime("%Y-%m-%d %H:%M:%S",time_struct)
    endtime = endtime / 1000
    time_struct = time.localtime(endtime)
    endtime = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)

    # 修改extend 和offest中整数型特征值的类型 （object—> int）
    extlist = ["ext1", "ext2", "ext3", "ext4", "ext5", "ext6",
                     "ext7", "ext8", "ext9", "ext10", "ext11", "ext12", "ext13", "ext14", "ext15", "ext16", "ext17",
                     "ext18", "ext19", "ext20", "ext21", "ext22", "ext23", "ext24"]
    extenddataframe[extlist] = extenddataframe[extlist].astype(int)
    offestdataframe[['off1','off2']] = offestdataframe[['off1','off2']].astype(int)


    #累加jobid的值
    # mdsdataframe = mdsdataframe.groupby(['jobid'],as_index=True).sum()
    # extenddataframe = extenddataframe.groupby(['jobid'],as_index=True).sum()
    # offestdataframe = offestdataframe.groupby(['jobid'],as_index=True).sum()

    mdsdataframe.rename(columns={'host': 'mdshost'},inplace=True)
    extenddataframe.rename(columns={'host': 'extendhost'},inplace=True)
    offestdataframe.rename(columns={'host': 'offesthost'},inplace=True)

    mdslist = ["mdshost", "rename", "setattr", "getattr", "statfs", "mkdir",
                     "getxattr", "sync", "setxattr", "mknod", "link", "rmdir", "samedir_rename", "close", "unlink",
                     "open", "crossdir_rename"]
    mdsdic = {i:'sum' for i in mdslist}

    extendlist = ["extendhost", "ext1", "ext2", "ext3", "ext4", "ext5", "ext6",
     "ext7", "ext8", "ext9", "ext10", "ext11", "ext12", "ext13", "ext14", "ext15", "ext16", "ext17",
     "ext18", "ext19", "ext20", "ext21", "ext22", "ext23", "ext24"]
    extenddic = {i:'sum' for i in extendlist}

    offestlist = ["offesthost", "fsname", "jobid", "off1", "off2"]
    offestdic = {i:'sum' for i in offestlist}

    mdsdataframe = mdsdataframe.groupby(['jobid'], as_index=True).agg(mdsdic)
    extenddataframe = extenddataframe.groupby(['jobid'], as_index=True).agg(extenddic)
    offestdataframe = offestdataframe.groupby(['jobid'], as_index=True).agg(offestdic)
    df = mdsdataframe.join(other=[extenddataframe,offestdataframe],how='outer')
    jobidlist = df.index.values.tolist()
    leng = len(jobidlist)
    df['jobid'] = jobidlist
    df['starttime'] = starttime
    df['endtime'] = endtime
    df.fillna(0,inplace=True)

    return df

def create_alldata(start,end,delta=5,windows=5):
    """
    :param start:int timestamp 训练样本/预测样本的开始时间
    :param end:int 训练样本/预测样本的结束时间
    :param delta:int(minutes) 间隔
    :param windows:int(minutes) 时间窗口
    :return:
    创建训练样本或预测样本
    """
    #将delta和windows转化为毫秒级，用于时间戳的加减
    delta = delta * 60 * 1000
    windows = windows * 60 * 1000
    ltime = start
    dflist = []
    while (ltime+windows<=end):
        starttime = ltime
        endtime = starttime + windows
        offestquery_json = offest_queryjson(starttime=starttime, endtime=endtime)
        extend_query_json = extend_queryjson(starttime=starttime, endtime=endtime)
        mds_query_json = mds_queryjson(starttime=starttime, endtime=endtime)
        offestmdata = search_bulk(index='search_lustreclientactionoffset', query_json=offestquery_json)
        offestdataframe = mdata_dataframe(mdata=offestmdata)
        extendmdata = search_bulk(index='search_lustreclientactionextent', query_json=extend_query_json)
        extenddataframe = mdata_dataframe(mdata=extendmdata)
        mdsmdata = search_bulk(index='search_lustreclientactionmds', query_json=mds_query_json)
        mdsdataframe = mdata_dataframe(mdata=mdsmdata)
        dflist.append(createdata(mdsdataframe, extenddataframe, offestdataframe, starttime=starttime, endtime=endtime))
        ltime = ltime + delta
    df = pd.concat(dflist,axis=0)
    return df


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
    es = Elasticsearch(['esheader01.ihep.ac.cn'], http_auth=('elastic', 'mine09443'), timeout=3600)
    #配置需要的字段
    #客户端的offest 格式为ts:1569466201,fs_name:acfs,job_id:44637629.0,0,101
    # starttime = 1569837600000
    # endtime = 1569837605000
    # offestquery_json = offest_queryjson(starttime=starttime,endtime=endtime)
    # extend_query_json = extend_queryjson(starttime=starttime, endtime=endtime)
    # mds_query_json = mds_queryjson(starttime=starttime,endtime=endtime)
    # offestmdata = search_bulk(index='search_lustreclientactionoffset',query_json=offestquery_json)
    # offestdataframe = mdata_dataframe(mdata=offestmdata)
    # offestdataframe.to_csv('D://IHEP/ioanomaly/offest.csv',index=False)
    # extendmdata = search_bulk(index='search_lustreclientactionextent',query_json=extend_query_json)
    # extenddataframe = mdata_dataframe(mdata=extendmdata)
    # extenddataframe.to_csv('D://IHEP/ioanomaly/extend.csv',index=False)
    # mdsmdata = search_bulk(index='search_lustreclientactionmds', query_json=mds_query_json)
    # mdsdataframe = mdata_dataframe(mdata=mdsmdata)
    # mdsdataframe.to_csv('D://IHEP/ioanomaly/mds.csv', index=False)
    # createdata(mdsdataframe,extenddataframe,offestdataframe,starttime=stattime,endtime=endtime)
    times = input('请输入采样的开始时间(%Y-%m-%d %H:%M:%S),结束时间(%Y-%m-%d %H:%M:%S),采样间隔(单位为分钟),时间窗口(单位为分钟)(以逗号间隔):')
    timelist = [i for i in times.split(',')]
    start = timelist[0]
    time_struct = time.strptime(start, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(time_struct)
    start = timestamp*1000
    end = timelist[1]
    time_struct = time.strptime(end, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(time_struct)
    end = timestamp * 1000
    delta = int(timelist[2])
    windows = int(timelist[3])
    df = create_alldata(start,end,delta,windows)
    df.to_csv('./traindata.csv', index=False)
    # flist = ["jobid", "off1", "off2", "rename", "setattr", "getattr", "statfs", "mkdir",
    #                  "getxattr", "sync", "setxattr", "mknod", "link", "rmdir", "samedir_rename", "close", "unlink",
    #                  "open", "crossdir_rename","ext1", "ext2", "ext3", "ext4", "ext5", "ext6",
    #                  "ext7", "ext8", "ext9", "ext10", "ext11", "ext12", "ext13", "ext14", "ext15", "ext16", "ext17",
    #                  "ext18", "ext19", "ext20", "ext21", "ext22", "ext23", "ext24"]
    flist = ["jobid", "mdshost","extendhost","offesthost","off1", "off2", "rename", "setattr", "getattr", "statfs", "mkdir",
             "getxattr", "sync", "setxattr", "mknod", "link", "rmdir", "samedir_rename", "close", "unlink",
             "open", "crossdir_rename", "ext1", "ext2", "ext3", "ext4", "ext5", "ext6",
             "ext7", "ext8", "ext9", "ext10", "ext11", "ext12", "ext13", "ext14", "ext15", "ext16", "ext17",
             "ext18", "ext19", "ext20", "ext21", "ext22", "ext23", "ext24"]
    flist.remove("jobid")
    flist.remove('mdshost')
    flist.remove('extendhost')
    flist.remove('offesthost')
    model = buildmodel()
    model = modelfit(model,dataframe=df,flist=flist)
    save(model,'./model.sav')

    times = input('请输入预测的开始时间(%Y-%m-%d %H:%M:%S),结束时间(%Y-%m-%d %H:%M:%S)（间隔时间和时间窗口同训练数据，若无结束时间，则，后不输入结束时间）:')
    timelist = [i for i in times.split(',')]
    start = timelist[0]
    time_struct = time.strptime(start, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(time_struct)
    start = timestamp * 1000
    end = timelist[1]
    #同时给出开始时间和结束时间
    if(end!=''):
        time_struct = time.strptime(end, "%Y-%m-%d %H:%M:%S")
        timestamp = time.mktime(time_struct)
        end = timestamp * 1000
        predictdf = create_alldata(start, end, delta, windows)
        res = modelpredict(model,predictdf,flist)
        res.to_csv('./predata.csv',index=False)
    #仅给出开始时间，则首先判断开始时间到当前时间的数据，再每5分钟判断一次
    else:
        end = time.localtime()
        timestamp = time.mktime(end)
        end = timestamp * 1000
        predictdf = create_alldata(start, end, delta, windows)
        res = modelpredict(model, predictdf, flist)
        res.to_csv('./predata.csv', index=False)
        start_ = time.localtime()
        timestamp = time.mktime(start_)
        start_ = timestamp * 1000
        end_ = timestamp * 1000 + 5 * 60 * 1000
        while(1):
            start = start_
            end = end_
            time.sleep(5*60)
            start_ = time.localtime()
            timestamp = time.mktime(start_)
            start_ = timestamp * 1000
            end_ = timestamp * 1000 + 5 * 60 * 1000
            predictdf = create_alldata(start, end, delta, windows)
            res = modelpredict(model, predictdf, flist)
            res.to_csv('./predata.csv',mode='a',header=False,index=False)







