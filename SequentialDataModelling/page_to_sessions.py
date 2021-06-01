#!/usr/bin/env python3
#SBATCH --partition=mcs.default.q
#SBATCH --output=openme.out

# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
word_vectorizer = CountVectorizer(ngram_range=(1,2), analyzer='word')

import numpy as np
from nltk import ngrams

import nltk
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


# In[2]:



df=pd.read_csv('./rawdata/BPI2016_Clicks_Logged_In.csv', sep=';', encoding='latin-1', keep_default_na=False)


# In[3]:


df['time'] = pd.to_datetime(df['TIMESTAMP'])
df['dates'] = df['time'].dt.date


# In[4]:


# pd.set_option("display.max_rows", None, "display.max_columns", None)
# df.head(100)


# In[5]:


qns=pd.read_csv('./rawdata/BPI2016_Questions.csv', sep=';', encoding='latin-1', keep_default_na=False)
comp=pd.read_csv('./rawdata/BPI2016_Complaints.csv', sep=';', encoding='latin-1', keep_default_na=False)
wrkmsg=pd.read_csv('./rawdata/BPI2016_Werkmap_Messages.csv', encoding='latin-1', sep=';', keep_default_na=False)


# In[6]:


# print(qns.columns)
# print(comp.columns)
# print(wrkmsg.columns)


# In[7]:


wm_columns_to_keep = ['EventDateTime','CustomerID','EventType']
# qns_columns_to_keep = ['ContactDate', 'ContactTimeStart', 'ContactTimeEnd','CustomerID']
qns_columns_to_keep = ['ContactDate', 'ContactTimeStart','CustomerID']
cmp_columns_to_keep = ['ContactDate','CustomerID']

wrkmsg=wrkmsg[[c for c in wrkmsg.columns if c in wm_columns_to_keep]]
wrkmsg.columns = ['CustomerID','ContactDate','EventType']
qns=qns[[c for c in qns.columns if c in qns_columns_to_keep]]
qns['EventType']='Customer question'
comp=comp[[c for c in comp.columns if c in cmp_columns_to_keep]]
comp['EventType']='Customer complaint'


# wrkmsg.head()
# 
# # wrkmsg['EventType'].unique()

# In[8]:


wrkmsg['ContactDate'] = pd.to_datetime(wrkmsg['ContactDate'])
wrkmsg['ContactTimeStart'] = [dt.datetime.time(d) for d in wrkmsg['ContactDate']] 
wrkmsg['ContactDate'] = [dt.datetime.date(d) for d in wrkmsg['ContactDate']] 

comp['ContactDate'] = pd.to_datetime(comp['ContactDate'])
# comp['ContactTimeStart'] = [dt.datetime.time(d) for d in comp['ContactDate']] 
comp['ContactTimeStart'] = '23:59:59'

cust_interactions = qns.append(comp, ignore_index=True)
cust_interactions = cust_interactions.append(wrkmsg, ignore_index=True)


# qns.head()

# comp.head()

# df.shape[0]*0.0015

# In[13]:





# In[14]:





# fdf.head(100)

# In[9]:


fdf=df[df['page_load_error']==0]
print(df.shape)
# fdf=df[df['page_load_error']==0]
print(fdf.shape)
# plerror_df=fdf[fdf['page_load_error']==1] #TBU - page load errors per page per session


# len(cust_interactions['CustomerID'].unique())

# # len(fdf['CustomerID'].unique())
# len(fdf[fdf['CustomerID'].isin(cust_interactions[cust_interactions['EventType']=='Werkmap message']['CustomerID'])]['CustomerID'].unique())

# In[10]:


#Subsetting to universe of customers with interactions
sessions_intr_customers=(fdf[fdf['CustomerID'].isin(cust_interactions['CustomerID'].unique())])
sessions_intr_customers.shape


# In[11]:


cust_interactions['dates']=cust_interactions['ContactDate']

session_interaction = pd.merge(sessions_intr_customers, cust_interactions, how="left", on=["CustomerID", "dates"])

session_interaction.shape #multiple customer interactions on same day


# print(len(fdf.PAGE_NAME.unique()))
# print(len(sessions_intr_customers.PAGE_NAME.unique()))
# print(len(session_interaction.PAGE_NAME.unique()))
# 
# # session_interaction.head(1000)

# In[12]:


session_interaction['Flag'] = np.where(pd.isnull(session_interaction['ContactDate'].values), 0, 1)
session_interaction.ContactDate = session_interaction.ContactDate.fillna('')
session_interaction.ContactTimeStart = session_interaction.ContactTimeStart.fillna('')
session_interaction.EventType = session_interaction.EventType.fillna('')
# session_interaction.head(1000)


# In[13]:


counts = session_interaction['PAGE_NAME'].value_counts()
# print (counts[:])
thres=session_interaction.shape[0]*0.01
cust_sess_intr=session_interaction[session_interaction['PAGE_NAME'].isin(counts[counts > thres].index)]
# print(len(cust_sess_intr.PAGE_NAME.unique()))
counts1 = cust_sess_intr['PAGE_NAME'].value_counts()
# print (counts1[:])


# session_interaction7 = session_interaction.loc[session_interaction["time"].dt.month == 7]

# session_interaction['time'] = pd.to_datetime(session_interaction['time'])
# 
# session_interaction7 = session_interaction.loc[session_interaction["time"].dt.month == 7]
# session_interaction8 = session_interaction.loc[session_interaction["time"].dt.month == 8]
# session_interaction9 = session_interaction.loc[session_interaction["time"].dt.month == 9]
# session_interaction10 = session_interaction.loc[session_interaction["time"].dt.month == 10]
# session_interaction11 = session_interaction.loc[session_interaction["time"].dt.month == 11]
# session_interaction12 = session_interaction.loc[session_interaction["time"].dt.month == 12]
# session_interaction1 = session_interaction.loc[session_interaction["time"].dt.month == 1]
# session_interaction2 = session_interaction.loc[session_interaction["time"].dt.month == 2]

# # session_qn1011 = session_qn10.append(session_qn11)
# # session_qn91011 = session_qn1011.append(session_qn9)
# return_df = pd.DataFrame(columns = session_interaction7.columns)
# session_level = session_interaction7.loc[session_interaction7['SessionID']==681]
# session_level['timespent']=session_level['time'].diff().apply(lambda x: x/np.timedelta64(1, 's')).fillna(0).astype('int64')
# return_df=return_df.append(session_level)

# In[14]:


# import datetime
def add_timespent(session_interaction):
    return_df = pd.DataFrame(columns = session_interaction.columns)
#     return_df=session_interaction
    unique_sess =  session_interaction['SessionID'].unique()
    for sess in unique_sess:
        session_level = session_interaction.loc[session_interaction['SessionID']==sess]
        session_level = session_level.sort_values(by=['CustomerID', 'TIMESTAMP'])
        session_level.loc[:,'timespent']=session_level.loc[:,'time'].diff().apply(lambda x: x/np.timedelta64(1, 's')).fillna(0).astype('int64')
        return_df=return_df.append(session_level,ignore_index=True)
#         print(return_df.head())
    return return_df


# In[15]:


def create_session_features(session_interaction):

    unique_sess =  session_interaction['SessionID'].unique()
    sess_feat_row = 0
    vect =  CountVectorizer(ngram_range = (2,2))
    for sess in unique_sess:
        sess_feat.at[sess_feat_row,'SessionID'] = sess
        session_level = session_interaction.loc[session_interaction['SessionID']==sess]
        session_pages = ""
        session_time = 0
        page_count = 0
        for index,row in session_level.iterrows():
            session_pages=session_pages+" "+session_level.at[index,'PAGE_NAME']
            page_count +=1
            session_time += session_level.at[index,'timespent']
        sess_feat.at[sess_feat_row,'twograms'] = str(session_pages)
        sess_feat.at[sess_feat_row,'page_count'] = page_count
        sess_feat.at[sess_feat_row,'session_time'] = session_time
        sess_feat.at[sess_feat_row,'avg_time_per_page'] = session_time/page_count
        sess_feat_row += 1
    
    bigram_df = pd.DataFrame(vect.fit_transform(sess_feat['twograms'].values).todense(), columns = vect.get_feature_names())

    final_df = pd.concat([sess_feat,bigram_df], axis=1)
    return(final_df)


# len(cust_sess_intr['SessionID'].unique())

# #CReatoing smaller subset

# Creating subset for feature engineering

# In[17]:


column_names = ["SessionID", "page_count","session_time","avg_time_per_page","twograms"]
sess_feat = pd.DataFrame(columns = column_names)


# In[21]:


def pre_process_and_export(major_sub,ind):

    start_time = time.time()
    sub = add_timespent(major_sub)
#     print("--- %s seconds ---" % (time.time() - start_time))


    

    start_time = time.time()
    major_sub_sess1 = create_session_features(sub)
#     print("--- %s seconds ---" % (time.time() - start_time))

    #Features - 'mean_timespent_per_page':'total_timespent_per_page'
    session_page=(sub.groupby(['SessionID','PAGE_NAME'])).agg(mean_timespent_per_page=('timespent', 'mean'), total_timespent_per_page=('timespent', 'sum')).reset_index()

    sess_page = pd.pivot(data=session_page,index='SessionID',columns='PAGE_NAME'
                         ,values=['mean_timespent_per_page','total_timespent_per_page'])
    sess_page.columns = sess_page.columns.map('|'.join).str.strip('|')
    sess_page=sess_page.reset_index().fillna(0)
    # sess_page.head()

    session_with_flags = pd.merge(major_sub_sess1, sess_page, how="left", on=["SessionID"])
    # session_with_flags.shape

    cust_info = sub.loc[:,['SessionID','CustomerID','AgeCategory','Gender','Office_U','Office_W']].drop_duplicates()

    sess_flag = sub.loc[:,['SessionID','Flag']].drop_duplicates()

    # cust_info.head()

    session_with_flags = pd.merge(session_with_flags, cust_info, how="left", on=["SessionID"])
    session_with_flags = pd.merge(session_with_flags, sess_flag, how="left", on=["SessionID"])

    session_with_flags.to_csv('./PyOut/session_level_ppcrsd'+str(ind)+'.csv', sep=',',index=False)
    print(session_with_flags.dtypes)
    print("--- %s seconds ---" % (time.time() - start_time))


# In[24]:





# In[25]:


import time
n_cust = 200
okunique_cust =  cust_sess_intr['CustomerID'].unique()[:n_cust]
all_cust =  cust_sess_intr['CustomerID'].unique()
okunique_cust=all_cust
major_sub = cust_sess_intr[cust_sess_intr['CustomerID'].isin(okunique_cust)]

print(len(major_sub.SessionID.unique()))
print(len(major_sub.TIMESTAMP.unique()))

start_time = time.time()


# sub = add_timespent(major_sub)
# pre_process_and_export(major_sub,"f200")
pre_process_and_export(major_sub,"all")


print("--- %s seconds ---" % (time.time() - start_time))
# sub = add_timespent(session_interaction7)


# In[ ]:


sess_feat.dtypes


# print(len(major_sub[major_sub.Flag==1].SessionID.unique())) #Sessions with flag =1 
# print(len(major_sub.SessionID.unique())) #total sessions

# # sub1=session_interaction7.head(1000)
# # %timeit sub = add_timespent(major_sub)
# 
# start_time = time.time()
# 
# 
# # sub = add_timespent(major_sub)
# pre_process_and_export(major_sub,"f100")
# 
# print("--- %s seconds ---" % (time.time() - start_time))
# # sub = add_timespent(session_interaction7)

# def create_sub_n1_n2_unique_customers(cust_sess_intr,n1,n2):
#     start_time = time.time()
#     okunique_cust =  cust_sess_intr['CustomerID'].unique()[n1:n2]
# 
#     major_sub = session_interaction[session_interaction['CustomerID'].isin(okunique_cust)]
#     print("--- %s seconds ---" % (time.time() - start_time))
#     return major_sub

# tempsub = create_sub_n1_n2_unique_customers(1,1000)
# 
# print(len(tempsub[tempsub.Flag==1].SessionID.unique())) #Sessions with flag =1 
# print(len(tempsub.SessionID.unique())) #total sessions

# In[26]:


print(len(cust_sess_intr[cust_sess_intr.Flag==1].SessionID.unique())) #Sessions with flag =1 
print(len(cust_sess_intr.SessionID.unique())) #total sessions


# # # sub.tail()
# # len(major_sub.SessionID.unique())
# 
# pre_process_and_export(tempsub,"f1000")

# In[28]:


# print(session_with_flags.dtypes[session_with_flags.dtypes=='object'])


# import math
# 
# n_cust = len(cust_sess_intr['CustomerID'].unique())
# 
# batch_size = 100 #in terms of unqiue customers
# 
# n_runs = math.ceil(n_cust / batch_size)
# 
# for i in range(n_runs):
#     print("Time for batch "+str(i))
#     ll = (i)*batch_size + 1
#     up = (i+1)*batch_size
#     tempsub = create_sub_n1_n2_unique_customers(cust_sess_intr,ll,up)
#     pre_process_and_export(tempsub,"_batch_"+str(i))

# n_list=[100,200,250,300,350,400,500]
#  
#     
# for n in n_list:
#     print("Time for first "+str(n)+" customers")
#     tempsub = create_sub_n1_n2_unique_customers(1,n)
#     pre_process_and_export(tempsub,"f"+str(n))

# 

# # sub.PAGE_NAME.unique()
# # sub.shape

# session_with_flags.to_csv('./PyOut/session_level_ppcrsd.csv', sep=',',index=False)

# # !pip install nltk
# 
# #DEPRECATED CODE
# 
# # def create_session_features(session_interaction):
#     
# 
# # #     unique_sess =  list(session_interaction['SessionID'].unique())
# #     unique_sess =  session_interaction['SessionID'].unique()
# #     sess_feat_row = 0
# #     vect =  CountVectorizer(ngram_range = (2,2))
# #     for sess in unique_sess:
# # #         sess_feat.SessionID = sess
# #         sess_feat.loc[sess_feat_row,'SessionID'] = sess
# #         session_level = session_interaction.loc[session_interaction['SessionID']==sess]
# #         session_pages = ""
# #         session_time = 0
# #         page_count = 0
# #         for index,row in session_level.iterrows():
# #             session_pages=session_pages+" "+session_level.loc[index,'PAGE_NAME']
# #             page_count +=1
# #             session_time += session_level.loc[index,'timespent']
# #         sess_feat.loc[sess_feat_row,'twograms'] = str(session_pages)
# #         sess_feat.loc[sess_feat_row,'page_count'] = page_count
# #         sess_feat.loc[sess_feat_row,'session_time'] = session_time
# #         sess_feat.loc[sess_feat_row,'avg_time_per_page'] = session_time/page_count
# #         sess_feat_row += 1
# 
# #     bigram_df = pd.DataFrame(vect.fit_transform(sess_feat['twograms'].values).todense(), columns = vect.get_feature_names())
# # #         bigram_df = pd.DataFrame(vect.fit_transform(session_pages).todense(), columns = vect.get_feature_names())
# # #         print([sess_feat.loc[:,'SessionID'],bigram_df])
# #     final_df = pd.concat([sess_feat,bigram_df], axis=1)
#         
# # #     print(sess_feat.shape)
# # #     return (pd.concat([sess_feat,bigram_df], axis=1))
# 
# # #     return(sess_feat)
# #     return(final_df)
# 

# sub = add_timespent(session_interaction7)
# sub=sub.append(add_timespent(session_interaction8))
# sub=sub.append(add_timespent(session_interaction9))
# sub=sub.append(add_timespent(session_interaction10))
# sub=sub.append(add_timespent(session_interaction11))
# sub=sub.append(add_timespent(session_interaction12))
# sub=sub.append(add_timespent(session_interaction1))
# sub=sub.append(add_timespent(session_interaction2))
# sub.to_csv(r'C:\Users\ishaa\Documents\Thesis\01 Data\PyOut\backup.csv', sep=',',index=False)

# sub.append(add_timespent(session_interaction8))

# session_interaction7.head()

# session_interaction.to_csv(r'C:\Users\ishaa\Documents\Thesis\01 Data\tmp\tmp.csv', sep=',',index=False)

# import site; site.getsitepackages()
