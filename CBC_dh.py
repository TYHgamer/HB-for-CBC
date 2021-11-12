# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:10:48 2021

@author: TYH
"""
import pandas as pd
import numpy as np

#data宽数据，alt个体长数据对应备选项属性，N受访者数量
#n_set选择集数量，n_alt备选项数量，ind_features消费者个人特征（列表）
def CBC_dh_tolong(data,alt,N,n_set,n_alt,ind_features=[]):
    #创建Id列
    Id = np.repeat([i+1 for i in range(0,N)],n_set*n_alt)
    data_Id = pd.DataFrame(Id)
    
    #创建选择集列
    set_id = []
    n = list(np.repeat([i+1 for i in range(0,n_set)],n_alt))
    for i in range(0,N):
        set_id = set_id+n
    set_id = pd.DataFrame(set_id)
    
    #创建备选项列
    alt_id = []
    m = [i+1 for i in range(0,n_alt)]
    for j in range(0,N*n_set):
        alt_id = alt_id+m
    alt_id = pd.DataFrame(alt_id)
    
    #创建对照选择列
    wide_choice = []
    for i in range(0,N):
        wide_choice.append(data.iloc[i,-n_set:].values)
    wc = np.array(wide_choice).ravel()
    wc = pd.DataFrame(list(np.repeat(wc,n_alt)))
    
    #选取个人特征
    data_ft = pd.DataFrame(np.repeat(data[ind_features].values,n_set*n_alt,axis=0))
    
    #拼接data_s
    data_s = pd.concat([data_Id,set_id,alt_id,wc,data_ft],axis=1)
    data_s.columns = ['id','set_id','alt_id','wide_choice']+ind_features
    
    data_s.loc[data_s['alt_id']==data_s['wide_choice'],'is_chosen'] =int(1)
    data_s.fillna(0,inplace=True)
    data_s['is_chosen'] = data_s['is_chosen'].astype('int')
    data_s.drop('wide_choice',axis=1,inplace=True)
    
    data_s = data_s[['id', 'set_id', 'alt_id', 'is_chosen']+ind_features]
    
    #加上备选项属性
    data_alt = pd.DataFrame()
    for i in range(0,N):
        data_alt = pd.concat([data_alt,alt],ignore_index=True)
    
    data_s = pd.concat([data_s,data_alt],axis=1)
    
    return data_s



#生成索引
def createidxs(n=0,N=0,names=[]):
    idxs = []
    n=0;m=N
    for i in range(0,len(names)):
        idxs.append((n,m))
        n = n+N
        m = m+N
    return idxs
    
def extractbeta_list(idxs,data):
    beta_list = []
    for i,j in idxs:
    #print(i,j)
        draft = []
        for n in range(i,j):
            draft.append(data['summary.mean'][n])
        beta_list.append(draft)
    return beta_list

def extractbeta(idxs,data):
    beta_result = []
    for i,j in idxs:
        #print(i,j)
        draft = []
        for n in range(i,j):
            draft.append(data['summary.mean'][n])
        beta_result.append(np.mean(draft))
    return beta_result

def extractstd_list(idxs,data):
    std_list = []
    for i,j in idxs:
    #print(i,j)
        draft = []
        for n in range(i,j):
            draft.append(data['summary.std'][n])
        std_list.append(draft)
    return std_list

def extractstd(idxs,data):
    std_result = []
    for i,j in idxs:
        #print(i,j)
        draft = []
        for n in range(i,j):
            draft.append(data['summary.std'][n])
        std_result.append(np.mean(draft))
    return std_result



#data第一列为参数(id,变量顺序)，第二列列名summary.mean,第二列列名summary.std
def stat_result(data,N,names):
    from scipy.stats import ttest_1samp as ttest
    from scipy.stats import sem
    
    idxs = createidxs(n=0,N=N,names=names)
    
    #计算beta均值及统计指标
    beta_list = extractbeta_list(idxs,data)
    beta_value = pd.DataFrame(zip(names,extractbeta(idxs,data=data)),columns=['parameters','mean']).round(3)
    beta_std_d = pd.DataFrame([np.std(beta_list[i]) for i in range(0,len(names))],columns=['mean_std_d'])
    beta_std_e = pd.DataFrame([sem(beta_list[i]) for i in range(0,len(names))],columns=['mean_std_e'])
    
    pvalue_list = [ttest(beta_list[i],0) for i in range(0,len(names))]
    beta_pvalue = pd.DataFrame(pvalue_list,columns=['t_value','p_value']).round(3)
    
    #计算beta抽样标准差均值及统计指标
    std_list = extractstd_list(idxs,data)
    std_paras = pd.DataFrame(extractstd(idxs,data=data),columns=['sd(paras)']).round(3)
    std_pvalue_list = [ttest(std_list[i],0) for i in range(0,len(names))]
    std_pvalue = pd.DataFrame(std_pvalue_list,columns=['std_tvalue','std_pvalue']).round(3)
    

    statistical_result = pd.concat([beta_value,beta_std_e,beta_std_d,beta_pvalue,std_paras,std_pvalue],axis=1)
    
    return  statistical_result

#提取不同符号beta的个数
def extractbeta_difs(data,N,names):
    idxs = createidxs(n=0,N=N,names=names)
    difs_n = []
    difs_m = []
    for i in range(0,len(names)):
        n = 0; m = 0
        for j in extractbeta_list(idxs,data)[i]:
            if j>0:
                n = n+1
            elif j<0:
                m = m+1
            else:
                print(f'存在等于0的beta,是第{i+1}个beta里的第{j+1}个')
        difs_n.append(n)
        difs_m.append(m)
    beta_difs = pd.concat([pd.DataFrame(names),pd.DataFrame(difs_n),pd.DataFrame(difs_m)],
                          axis=1)
    beta_difs.columns = ['parameters','正','负']
    
    return beta_difs
