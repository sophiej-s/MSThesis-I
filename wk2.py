#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 22:36:18 2022

@author: sophie
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

path_to_file='/Users/sophie/Desktop/Recommenders/CSE158/4/amazon_reviews_us_Toys_v1_00.tsv'

# data = pd.read_csv(path_to_file, delim_whitespace=True)



# df = pd.read_csv(path_to_file,sep='\t')

# df = pd.read_csv(path_to_file)


# df = pd.read_csv(path_to_file, sep="\t", header=None)

# data=pd.read_table(path_to_file)


df=pd.read_csv(path_to_file, sep="\t", header=0, error_bad_lines=False) #ignore some errors
#%%

df[["star_rating"]].describe()

#         star_rating
# count  4.859606e+06
# mean   4.211690e+00
# std    1.263354e+00
# min    1.000000e+00
# 25%    4.000000e+00
# 50%    5.000000e+00
# 75%    5.000000e+00
# max    5.000000e+00



df.count(axis=0, level=None, numeric_only=False)#  Count non-NA cells for each column or row.
# marketplace          4859607
# customer_id          4859607
# review_id            4859607
# product_id           4859607
# product_parent       4859607
# product_title        4859561
# product_category     4859607
# star_rating          4859606
# helpful_votes        4859606
# total_votes          4859606
# vine                 4859606
# verified_purchase    4859606
# review_headline      4859575
# review_body          4859234
# review_date          4859578





#drop the rows where 
df=df.dropna(subset=['review_headline', 'review_body', 'star_rating'])

df.count(axis=0, level=None, numeric_only=False)#  Count non-NA cells for each column or row.

# marketplace          4859203
# customer_id          4859203
# review_id            4859203
# product_id           4859203
# product_parent       4859203
# product_title        4859157
# product_category     4859203
# star_rating          4859203
# helpful_votes        4859203
# total_votes          4859203
# vine                 4859203
# verified_purchase    4859203
# review_headline      4859203
# review_body          4859203
# review_date          4859175


df['star_rating'].value_counts()

# 5.0    3073553
# 4.0     769076
# 1.0     399076
# 3.0     387341
# 2.0     230157


#%% SENTIMENT analysis
#for the purpose of recommending products, assume reviews with rating 4,5 are positve sentiment (1); 3 and below are negative (0)


df.dtypes

# marketplace           object
# customer_id            int64
# review_id             object
# product_id            object
# product_parent         int64
# product_title         object
# product_category      object
# star_rating          float64
# helpful_votes        float64
# total_votes          float64
# vine                  object
# verified_purchase     object
# review_headline       object
# review_body           object
# review_date           object



df['sentiment']=np.where(df['star_rating']>=4, 1, 0)





df['sentiment'].value_counts()
# 1    3842629
# 0    1016574


#sort by helpful
df=df.sort_values(by='helpful_votes', ascending=False)

#NOT SHUFFLING
#df_shuffled=df.sample(frac=1, replace=False) #without replacement; keep fraction is all  or 100%
df_shuffled=df
#select 10K  first samples of  each sentiment 

df_sample1=df_shuffled[df_shuffled['sentiment']==1][0:1000]
df_sample0=df_shuffled[df_shuffled['sentiment']==0][0:1000]


# A=(df.head(5))


df_sample=df_sample1.append(df_sample0)


#%%  is there a correlation between the rating number , helpful votes and the the length  of the review?

df['review_body_count']=np.zeros(([df['review_body'].count(),1]))

df['review_body_count']=df['review_body_count'].astype('int')

for i  in range(0, df['review_body'].count()-1):
   df.loc[i, ('review_body_count')]= len(df.iloc[i,:]['review_body'])
   

all_corr=df.corr(method ='pearson')
   



#%% text processing


nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english"))
from nltk.stem import PorterStemmer

ps = PorterStemmer()


#remove numerals in the reviews.


reviews= pd.DataFrame()
reviews['review_body']=df_sample['review_body']
reviews['sentiment']=df_sample['sentiment']
reviews= reviews.reset_index()



for i  in range(0, reviews['review_body'].count()):
   #reviews.loc[i, ('review_body')]= reviews.loc[i, ('review_body')].replace("<br />", " ")
   tokens= word_tokenize(reviews.loc[i, ('review_body')])
   tokens = [w.lower()  for w in tokens ]
   tokens = [w for w in tokens if not w in stop_words]
   tokens = [w for w in tokens if w.isalpha()]
   #tokens = [ps.stem(w) for w in tokens]
   reviews.loc[i, ('review_body')]=' '.join(tokens)
   


# ' '.join(tokens)
# Out[214]: 'littl disappoint first bought item function limit year old son point passeng shoe remov place deadli fingernail file underneath passeng scarf neither detector doorway secur wand pick son said worst secur ever turn okay passeng got playmobil tri hijack mob coupl heroic passeng sustain minor injuri scuffl treat playmobil hospit best thing product teach kid realiti live societi son said want playmobil neighborhood surveil system set christma heard cc tv camera thing pretti worthless term qualiti motion detect think get playmobil interog set instead come cute littl memo georg bush'

# reviews.loc[i, ('review_body')]
# Out[215]: 'I was a little disappointed when I first bought this item, because the functionality is limited.  My 5 year old son pointed out that the passenger\'s shoes cannot be removed.  Then, we placed a deadly fingernail file underneath the passenger\'s scarf, and neither the detector doorway nor the security wand picked it up.  My son said \\\\"that\'s the worst security ever!\\\\".  But it turned out to be okay, because when the passenger got on the Playmobil B757 and tried to hijack it, she was mobbed by a couple of other heroic passengers, who only sustained minor injuries in the scuffle, which were treated at the Playmobil Hospital.    The best thing about this product is that it teaches kids about the realities of living in a high-surveillence society.  My son said he wants the Playmobil Neighborhood Surveillence System set for Christmas.  I\'ve heard that the CC TV cameras on that thing are pretty worthless in terms of quality and motion detection, so I think I\'ll get him the Playmobil Abu-Gharib Interogation Set instead (it comes with a cute little memo from George Bush).'





#%%







#%%
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline



X_train, X_test, y_train, y_test = train_test_split(reviews['review_body'], reviews['sentiment'], test_size=0.3)





from sklearn.feature_extraction.text import TfidfVectorizer
T_vectorizer=TfidfVectorizer()

tf_X_train=T_vectorizer.fit_transform(X_train)
tf_X_test=T_vectorizer.transform(X_test)

Logistic_clfr=LogisticRegression()

Logistic_clfr.fit(tf_X_train, y_train)
y_pred=Logistic_clfr.predict(tf_X_test)


#no parameter tuning
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=report#sklearn.metrics.classification_report
from sklearn.metrics import classification_report
target_names = ['0 = negative (rating lower than 3)',  '1 = positive (rating 4 or 5)'] # 0 = negative, 4 = positive
print(classification_report(y_test, y_pred, target_names=target_names))


#Results on 2K data samples
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.76      0.68      0.72       307
#       1 = positive (rating 4 or 5)       0.70      0.77      0.73       293

#                           accuracy                           0.73       600
#                          macro avg       0.73      0.73      0.73       600
#                       weighted avg       0.73      0.73      0.73       600




from sklearn.metrics import  confusion_matrix
cnf_matrix1 = confusion_matrix(y_test,y_pred,labels=[0,1])

print(cnf_matrix1)
# [[210  97]
#  [ 67 226]]

y_test.value_counts()
# 0    307
# 1    293






Tf_idf__vectorizer=TfidfVectorizer(use_idf=True)#,max_features=10)
tf_X_train=T_vectorizer.fit_transform(X_train)
tf_X_test=T_vectorizer.transform(X_test)

Logistic_clfr=LogisticRegression()

Logistic_clfr.fit(tf_X_train, y_train)
y_pred=Logistic_clfr.predict(tf_X_test)


#no parameter tuning
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=report#sklearn.metrics.classification_report
from sklearn.metrics import classification_report
target_names = ['0 = negative (rating lower than 3)',  '1 = positive (rating 4 or 5)'] # 0 = negative, 4 = positive
print(classification_report(y_test, y_pred, target_names=target_names))




# 0 = negative (rating lower than 3)       0.76      0.68      0.72       307
#       1 = positive (rating 4 or 5)       0.70      0.77      0.73       293

#                           accuracy                           0.73       600
#                          macro avg       0.73      0.73      0.73       600
#                       weighted avg       0.73      0.73      0.73       600









