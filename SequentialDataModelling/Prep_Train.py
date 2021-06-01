#!/usr/bin/env python
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

import sys
sys.path.insert(0, r"c:\users\ishaa\anaconda3\envs\tf_gpu\lib\site-packages")

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers

# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from keras.optimizers import SGD
from keras import Sequential

# Set random seed
np.random.seed(21)


# In[2]:


#df=pd.read_csv(r'C:\Users\ishaa\Documents\Thesis\01 Data\BPI16\12674816\BPI2016_Clicks_Logged_In.csv', sep=';', encoding='latin-1', keep_default_na=False)
df=pd.read_csv('./rawdata/BPI2016_Clicks_Logged_In.csv', sep=';', encoding='latin-1', keep_default_na=False)

# In[3]:


df['time'] = pd.to_datetime(df['TIMESTAMP'])
df['dates'] = df['time'].dt.date


# In[4]:


# pd.set_option("display.max_rows", None, "display.max_columns", None)
df.head(100)


# In[5]:


#qns=pd.read_csv(r'C:\Users\ishaa\Documents\Thesis\01 Data\BPI16\BPI2016_Questions.csv', sep=';', encoding='latin-1', keep_default_na=False)
#comp=pd.read_csv(r'C:\Users\ishaa\Documents\Thesis\01 Data\BPI16\BPI2016_Complaints.csv', sep=';', encoding='latin-1', keep_default_na=False)
#wrkmsg=pd.read_csv(r'C:\Users\ishaa\Documents\Thesis\01 Data\BPI16\BPI2016_Werkmap_Messages.csv', encoding='latin-1', sep=';', keep_default_na=False)
qns=pd.read_csv('./rawdata/BPI2016_Questions.csv', sep=';', encoding='latin-1', keep_default_na=False)
comp=pd.read_csv('./rawdata/BPI2016_Complaints.csv', sep=';', encoding='latin-1', keep_default_na=False)
wrkmsg=pd.read_csv('./rawdata/BPI2016_Werkmap_Messages.csv', encoding='latin-1', sep=';', keep_default_na=False)


# In[6]:


print(qns.columns)
print(comp.columns)
print(wrkmsg.columns)


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


# In[12]:


print(len(fdf.PAGE_NAME.unique()))
print(len(sessions_intr_customers.PAGE_NAME.unique()))
print(len(session_interaction.PAGE_NAME.unique()))

# session_interaction.head(1000)


# In[13]:


session_interaction['Flag'] = np.where(pd.isnull(session_interaction['ContactDate'].values), 0, 1)
session_interaction.ContactDate = session_interaction.ContactDate.fillna('')
session_interaction.ContactTimeStart = session_interaction.ContactTimeStart.fillna('')
session_interaction.EventType = session_interaction.EventType.fillna('')
# session_interaction.head(1000)


# In[14]:


counts = session_interaction['PAGE_NAME'].value_counts()
# print (counts[:])
thres=session_interaction.shape[0]*0.01
cust_sess_intr=session_interaction[session_interaction['PAGE_NAME'].isin(counts[counts > thres].index)]
print(len(cust_sess_intr.PAGE_NAME.unique()))
counts1 = cust_sess_intr['PAGE_NAME'].value_counts()
print (counts1[:])


# session_interaction7 = session_interaction.loc[session_interaction["time"].dt.month == 7]

# In[15]:


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


# In[16]:


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


# In[17]:


len(cust_sess_intr['SessionID'].unique())


# #CReatoing smaller subset

# Creating subset for feature engineering

# In[18]:


column_names = ["SessionID", "page_count","session_time","avg_time_per_page","twograms"]
sess_feat = pd.DataFrame(columns = column_names)


# In[19]:


def pre_process_and_return(major_sub,ind):

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
#     print(session_with_flags.dtypes)
    print("--- %s seconds ---" % (time.time() - start_time))
    return session_with_flags


# In[20]:


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
#     print(session_with_flags.dtypes)
    print("--- %s seconds ---" % (time.time() - start_time))


# In[21]:


def create_sub_n1_n2_unique_customers(cust_sess_intr,n1,n2):
    start_time = time.time()
    okunique_cust =  cust_sess_intr['CustomerID'].unique()[n1:n2]

    major_sub = session_interaction[session_interaction['CustomerID'].isin(okunique_cust)]
    print("--- %s seconds ---" % (time.time() - start_time))
    return major_sub


# In[22]:


# import time
n_cust = 100

all_cust =  cust_sess_intr['CustomerID'].unique()
okunique_cust=all_cust

# okunique_cust =  list(cust_sess_intr['CustomerID'].unique()[:n_cust])
major_sub = cust_sess_intr[cust_sess_intr['CustomerID'].isin(okunique_cust)]

# major_sub = create_sub_n1_n2_unique_customers(cust_sess_intr,1200,1200+n_cust)


print(len(major_sub.SessionID.unique()))
print(len(major_sub.TIMESTAMP.unique()))

start_time = time.time()


# sub = add_timespent(major_sub)
# pre_process_and_export(major_sub,"f200")

sess_level = pre_process_and_return(major_sub,"all")

print("--- %s seconds ---" % (time.time() - start_time))
# sub = add_timespent(session_interaction7)


# In[23]:


# sess_level = pre_process_and_return(major_sub,"r100")


# In[24]:


sess_level.dtypes


# In[25]:


print(len(major_sub[major_sub.Flag==1].SessionID.unique())) #Sessions with flag =1 
print(len(major_sub.SessionID.unique())) #total sessions


# ## Modelling starts here

# In[ ]:





# In[38]:


session_with_flags=sess_level

session_with_flags.loc[:,'twograms'] =  session_with_flags.loc[:,'twograms'].astype(str)
session_with_flags.loc[:,'CustomerID'] =  session_with_flags.loc[:,'CustomerID'].astype(str)
session_with_flags.loc[:,'Gender'] =  session_with_flags.loc[:,'Gender'].astype(str)
session_with_flags.loc[:,'Office_U'] =  session_with_flags.loc[:,'Office_U'].astype(str)
session_with_flags.loc[:,'Office_W'] =  session_with_flags.loc[:,'Office_W'].astype(str)
session_with_flags.loc[:,'Flag'] =  session_with_flags.loc[:,'Flag'].astype(str)

cols=session_with_flags.filter(regex=(".*timespent.*")).columns
# 
for c in cols:
    session_with_flags.loc[:,c] =  session_with_flags.loc[:,c].replace('',0.0)
    session_with_flags.loc[:,c] =  session_with_flags.loc[:,c].astype('float16')
#     del session_with_flags[c]
    print(c)
#     print(session_with_flags.loc[:,c].values == '').sum()


# In[39]:



encoder = LabelEncoder()
#  transfomed_label = encoder.fit_transform(["dog", "cat", "bird"])

# del (session_with_flags['SessionID'])
session_with_flags.loc[:,'twograms'] =  encoder.fit_transform(session_with_flags.loc[:,'twograms'])
# session_with_flags.loc[:,'twograms'] =  session_with_flags.loc[:,'twograms'].astype('float16')

# session_with_flags.loc[:,'CustomerID'] =  session_with_flags.loc[:,'CustomerID'].astype(str)
session_with_flags.loc[:,'CustomerID'] =  encoder.fit_transform(session_with_flags.loc[:,'CustomerID'])
# session_with_flags.loc[:,'CustomerID'] =  session_with_flags.loc[:,'CustomerID'].astype('float16')

session_with_flags.loc[:,'AgeCategory'] =  encoder.fit_transform(session_with_flags.loc[:,'AgeCategory'])
# session_with_flags.loc[:,'AgeCategory'] =  session_with_flags.loc[:,'AgeCategory'].astype('float16')

session_with_flags.loc[:,'Gender'] =  encoder.fit_transform(session_with_flags.loc[:,'Gender'])
# session_with_flags.loc[:,'Gender'] =  session_with_flags.loc[:,'Gender'].astype(int)

# session_with_flags.loc[:,'Office_U'] =  session_with_flags.loc[:,'Office_U'].astype(str)
session_with_flags.loc[:,'Office_U'] =  encoder.fit_transform(session_with_flags.loc[:,'Office_U'])
# session_with_flags.loc[:,'Office_U'] =  session_with_flags.loc[:,'Office_U'].astype('float16')

# session_with_flags.loc[:,'Office_W'] =  session_with_flags.loc[:,'Office_W'].astype(str)
session_with_flags.loc[:,'Office_W'] =  encoder.fit_transform(session_with_flags.loc[:,'Office_W'])
# session_with_flags.loc[:,'Office_W'] =  session_with_flags.loc[:,'Office_W'].astype('float16')

session_with_flags.loc[:,'Flag'] =  encoder.fit_transform(session_with_flags.loc[:,'Flag'])
# session_with_flags.loc[:,'Flag'] =  session_with_flags.loc[:,'Flag'].astype('float16')

print(session_with_flags.dtypes[session_with_flags.dtypes=='object'])


# In[40]:


session_with_flags.loc[:,'page_count'] =  session_with_flags.loc[:,'page_count'].astype('float16')
session_with_flags.loc[:,'session_time'] =  session_with_flags.loc[:,'page_count'].astype('float16')
session_with_flags.loc[:,'avg_time_per_page'] =  session_with_flags.loc[:,'avg_time_per_page'].astype('float16')

session_with_flags = session_with_flags.drop(['SessionID'], axis=1)


# In[41]:


session_with_flags.dtypes


# In[49]:


print(len(session_with_flags[session_with_flags.Flag==1]),len(session_with_flags)) #Sessions with flag =1 


# In[50]:


# session_with_flags.shape


# In[48]:


msk = np.random.rand(len(session_with_flags)) < 0.5
train = session_with_flags[msk]
val_test = session_with_flags[~msk]

msk = np.random.rand(len(val_test)) < 0.5
validate = val_test[msk]
test = val_test[~msk]

train_target = train.loc[:,"Flag"]
validate_target = validate.loc[:,"Flag"]
test_target = test.loc[:,"Flag"]

print(len(train[train.Flag==1]),len(train)) #Sessions with flag =1 
print(len(validate[validate.Flag==1]),len(validate)) #Sessions with flag =1 
print(len(test[test.Flag==1]),len(test)) #Sessions with flag =1 

train_data = train.drop(['Flag'], axis=1)
validate_data = validate.drop(['Flag'], axis=1)
test_data = test.drop(['Flag'], axis=1)


# In[ ]:


print(len(major_sub[major_sub.Flag==1].SessionID.unique())) #Sessions with flag =1 


# In[43]:




number_of_features = train_data.shape[1]

# # Start neural network
network = Sequential()

# # Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=8, activation='relu', input_shape=(number_of_features,)))

# network.add(LSTM(units=32,activation="tanh", dropout=0.0, recurrent_dropouat=0.0, use_bias=True, unroll=False))

# 

# # Add fully connected layer with a ReLU activation function
network.add(layers.Dense(units=250, activation='relu'))
network.add(layers.Dense(units=250, activation='relu'))
network.add(layers.Dense(units=250, activation='relu'))
network.add(layers.Dense(units=250, activation='relu'))

# # Add fully connected layer with a sigmoid activation function
network.add(layers.Dense(units=1, activation='sigmoid')) #output layer

# optimizer=Adadelta(lr=.1.5)

network.compile(loss='binary_crossentropy', # Cross-entropy
                optimizer='sgd', # Root Mean Square Propagation
#                    optimizer = SGD(learning_rate=0.01, momentum=0.1, nesterov=True), 
                metrics=['acc']) # Accuracy performance metric


#https://chrisalbon.com/deep_learning/keras/feedforward_neural_network_for_binary_classification/


# In[45]:


# train_features,test_features=prepare_inputs(enc,train_data,test_data)

history = network.fit(train_data, # Features
                      train_target, # Target vector
                      epochs=20, # Number of epochs
                      verbose=1, # Print description after each epoch
                      batch_size=512, # Number of observations per batch
                      
                      validation_data=(validate_data, validate_target)) # Data for evaluation


# In[ ]:


network.save('./PyOut/Models')

