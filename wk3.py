#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 17:28:36 2022

@author: sophie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk



#%% SECTION 1

path_to_file='amazon_reviews_us_Toys_v1_00.tsv'
df=pd.read_csv(path_to_file, sep="\t", header=0, error_bad_lines=False) #ignore some errors
df=df.dropna(subset=['review_headline', 'review_body', 'star_rating'])
df['sentiment']=np.where(df['star_rating']>=4, 1, 0)

df_shuffled=df.sample(frac=1, replace=False) #without replacement; keep fraction is all  or 100%

df['sentiment']=np.where(df['star_rating']>=4, 1, 0)
df['sentiment'].value_counts()
# 1    3,842,629
# 0    1,016,574


df_sample1=df_shuffled[df_shuffled['sentiment']==1][0:1000]
df_sample0=df_shuffled[df_shuffled['sentiment']==0][0:1000]
df_sample=df_sample1.append(df_sample0)


nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words("english"))
from nltk.stem import PorterStemmer

ps = PorterStemmer()


reviews= pd.DataFrame()
reviews['review_body']=df_sample['review_body']
reviews['sentiment']=df_sample['sentiment']
reviews= reviews.reset_index()


for i  in range(0, reviews['review_body'].count()):
   tokens= word_tokenize(reviews.loc[i, ('review_body')])
   tokens = [w.lower()  for w in tokens ]
   tokens = [w for w in tokens if not w in stop_words]
   tokens = [w for w in tokens if w.isalpha()]
   tokens = [ps.stem(w) for w in tokens]
   reviews.loc[i, ('review_body')]=' '.join(tokens)
   

#---------

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


X_train, X_test, y_train, y_test = train_test_split(reviews['review_body'], reviews['sentiment'], test_size=0.3)



from sklearn.feature_extraction.text import TfidfVectorizer
T_vectorizer=TfidfVectorizer()

tf_X_train=T_vectorizer.fit_transform(X_train)
tf_X_test=T_vectorizer.transform(X_test)

Logistic_clfr=LogisticRegression(max_iter=1000)

Logistic_clfr.fit(tf_X_train, y_train)
y_pred=Logistic_clfr.predict(tf_X_test)

y_pred_prob=Logistic_clfr.predict_proba(tf_X_test) #predict_proba(X)   Predict  probability estimates.


from sklearn.metrics import classification_report
target_names = ['0 = negative (rating lower than 3)',  '1 = positive (rating 4 or 5)'] # 0 = negative, 4 = positive
print(classification_report(y_test, y_pred, target_names=target_names))



from sklearn.metrics import  confusion_matrix
cnf_matrix1 = confusion_matrix(y_test,y_pred,labels=[0,1])

print(cnf_matrix1)


#%% OUTPUT

#  2K  total reviews NOT sorted by the most helpful and STEMMED.
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.72      0.68      0.70       302
#       1 = positive (rating 4 or 5)       0.70      0.73      0.71       298

#                           accuracy                           0.71       600
#                          macro avg       0.71      0.71      0.71       600
#                       weighted avg       0.71      0.71      0.71       600



# Re-Ran the 2K revies sample  NOT sorted by the most helpful and NOT STEMMED. 
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.71      0.70      0.70       309
#       1 = positive (rating 4 or 5)       0.68      0.70      0.69       291

#                           accuracy                           0.70       600
#                          macro avg       0.70      0.70      0.70       600
#                       weighted avg       0.70      0.70      0.70       600






#%%

#Expand the sample size to 10K per each sentiment. Stemmed. NOT sorted by the most helpful.

df_sample1=df_shuffled[df_shuffled['sentiment']==1][0:10000]
df_sample0=df_shuffled[df_shuffled['sentiment']==0][0:10000]
df_sample=df_sample1.append(df_sample0)


#Expand the sample size to 10K per each sentiment. Stemmed
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.83      0.79      0.81      3043
#       1 = positive (rating 4 or 5)       0.79      0.83      0.81      2957

#                           accuracy                           0.81      6000
#                          macro avg       0.81      0.81      0.81      6000
#                       weighted avg       0.81      0.81      0.81      6000


#The sample size is 10K per each sentiment. Not stemmed
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.81      0.80      0.81      2993
#       1 = positive (rating 4 or 5)       0.81      0.81      0.81      3007

#                           accuracy                           0.81      6000
#                          macro avg       0.81      0.81      0.81      6000
#                       weighted avg       0.81      0.81      0.81      6000


#%%


df_sample1=df_shuffled[df_shuffled['sentiment']==1][0:30000]
df_sample0=df_shuffled[df_shuffled['sentiment']==0][0:30000]
df_sample=df_sample1.append(df_sample0)

#Expand to 60K total reviews (even split), not sorted by the most helpful. Stemmed.

#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.85      0.82      0.83      9028
#       1 = positive (rating 4 or 5)       0.83      0.85      0.84      8972

#                           accuracy                           0.84     18000
#                          macro avg       0.84      0.84      0.84     18000
#                       weighted avg       0.84      0.84      0.84     18000


#Same 60K total reviews; not sorted by the most helpful. Not Stemmed.

# 0 = negative (rating lower than 3)       0.84      0.82      0.83      9072
#       1 = positive (rating 4 or 5)       0.82      0.85      0.84      8928

#                           accuracy                           0.83     18000
#                          macro avg       0.83      0.83      0.83     18000
#                       weighted avg       0.83      0.83      0.83     18000



#%%------
#Take 60 K total reviews organized by the most helpful, stemmed.
    
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.84      0.82      0.83      8986
#       1 = positive (rating 4 or 5)       0.82      0.84      0.83      9014

#                           accuracy                           0.83     18000
#                          macro avg       0.83      0.83      0.83     18000
#                       weighted avg       0.83      0.83      0.83     18000

# [[7370 1616]
#  [1448 7566]]



#Take 60 K total reviews organized by the most helpful, NOT stemmed.

#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.84      0.83      0.83      9018
#       1 = positive (rating 4 or 5)       0.83      0.84      0.84      8982

#                           accuracy                           0.84     18000
#                          macro avg       0.84      0.84      0.84     18000
#                       weighted avg       0.84      0.84      0.84     18000

# [[7460 1558]
#  [1394 7588]]


#Still organized by the helpfullness, take 600K in total.  Stemmed

df_sample1=df_shuffled[df_shuffled['sentiment']==1][0:300000]
df_sample0=df_shuffled[df_shuffled['sentiment']==0][0:300000]
df_sample=df_sample1.append(df_sample0)
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.86      0.86      0.86     89702
#       1 = positive (rating 4 or 5)       0.86      0.86      0.86     90298

#                           accuracy                           0.86    180000
#                          macro avg       0.86      0.86      0.86    180000
#                       weighted avg       0.86      0.86      0.86    180000

# [[77146 12556]
#  [12771 77527]]



#--------------------------------------------------------------------------------------------------------
#%%  SECTION 2
#Using word2vec
#Corpus of data is at https://code.google.com/archive/p/word2vec/
#The archive is available here: GoogleNews-vectors-negative300.bin.gz." 

from gensim.models import KeyedVectors
import gensim
import seaborn as sns
import numpy as np
import xgboost as xgb


#  Word2Vec Use the Google model (Wiki data set can be used to train another model = future work)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True) 

# Check dimension of word embeddings (should be 300 per the Google Specs)
word2vec_model.vector_size
# Out[9]: 300

word2vec_model.similar_by_word('toy', topn=20)
# [('toys', 0.79255610704422),
#  ('Toy', 0.6904610395431519),
#  ('Toys', 0.6202871203422546),
#  ('doll', 0.6188485026359558),
#  ('dolls', 0.6039299368858337),
#  ('Tickle_Me_Elmo_doll', 0.5959895849227905),
#  ('plush_toys', 0.5801015496253967),
#  ('toymaker', 0.5789716243743896),
#  ('Beyblades', 0.5757725834846497),
#  ('Hasbro_Inc_HAS.N', 0.5757586359977722),
#  ('toymakers', 0.5752304792404175),
#  ('Toys_R_Us', 0.5699595212936401),
#  ('retailer_Hamleys', 0.5690807104110718),
#  ('Tonka_truck', 0.5679948329925537),
#  ('Zhu_Zhu_pet', 0.5670112371444702),
#  ('stuffed_animal', 0.5664635300636292),
#  ('Playskool_toys', 0.5650117993354797),
#  ('Danish_toymaker', 0.5644108057022095),
#  ('playthings', 0.563970685005188),
#  ('Furbys', 0.561872661113739)]



#%%
df_sample = pd.DataFrame()

df_sample['review_body'] = pd.DataFrame(df['review_body']).sample(frac = 0.002, random_state= 200)

df_sample= df_sample.reset_index()

for i  in range(0, df_sample['review_body'].count()):
   tokens= word_tokenize(df_sample.loc[i, ('review_body')])
   tokens = [w.lower()  for w in tokens ]
   tokens = [w for w in tokens if not w in stop_words]
   tokens = [w for w in tokens if w.isalpha()]
   # tokens = [ps.stem(w) for w in tokens]   #NOT STEMMED

   #check if the words are in the word2vc vocabulary 
   tokens = [w for w in tokens if w in word2vec_model.key_to_index ]
    
   df_sample.loc[i, ('review_body')]=' '.join(tokens)


#word2vec contains stemmed words 
 # 'beauti littl set granddaught love'
#  'beauti' in word2vec_model.key_to_index
# Out[78]: True

# 'littl' in word2vec_model.key_to_index
# Out[79]: True




# Take a sub-set of reviews (9718) and take a mean representing each review by the mean of word embeddings for the words used in the review body.
word2vec_model_embeddings = WordVecVectorizer(word2vec_model)
word2vec_model_embeddings_ave = word2vec_model_embeddings.transform(df_sample['review_body'])
print(word2vec_model_embeddings_ave.shape)
# (97184, 300)


from sklearn.cluster import KMeans
km = KMeans( n_clusters=10, init='k-means++',  n_init=100, max_iter=500,random_state=0, algorithm="full")
y_km = km.fit_predict(word2vec_model_embeddings_ave)
df_sample['Topic_Cluster'] = pd.DataFrame(y_km)


#TSNE
from sklearn.manifold import TSNE
tsne=TSNE(n_components=3, init='random', perplexity=100) #perplexity = required neighbors,

#test on 8 rows 
word2vec_model_embeddings_ave[2:10,:].shape
# Out[88]: (8, 300)

# tsne_3D=tsne.fit_transform(word2vec_model_embeddings_ave[2:10, :])
# tsne_3D.shape
# Out[90]: (8, 3)

tsne_3D=tsne.fit_transform(word2vec_model_embeddings_ave)

tsne_3D.shape
# Out[232]: (9718, 3)
# tsne_3D_df=pd.DataFrame(tsne_3D)
# tsne_3D_df.to_csv("3D_Embeddings.tsv", sep = "\t",  header=True, index=False, index_label=None)
#projecting  embeddings at https://projector.tensorflow.org  


#plot the vectors in 3D:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tsne_3D[:,0], tsne_3D[:,1], tsne_3D[:,2], c=y_km, s=60)
ax.view_init(30, 185)
plt.show()

df_sample['Topic_Cluster'].value_counts()
# Out[107]: 
# 0    5263
# 8    2199
# 6     994
# 7     625
# 5     238
# 2     169
# 4      90
# 9      61
# 3      42
# 1      37



#Rank the most popular words in each cluster to determine the topics

def cluster_topic(df_sample, topic_number=1):
    df_filtered = df_sample.pipe(lambda x: x.loc[x['Topic_Cluster'] == topic_number])
    #stem and the words 
    
    list_reviews= ( [ ps.stem(word) for word in df_filtered['review_body'] ])
    
    #count freq of words in the reviews in each cluester
    from collections import Counter
    count = dict(Counter(word for sentence in list_reviews for word in sentence.split()))
    
    
    sorted_count = dict( sorted(count.items(),   key=lambda item: item[1], reverse=True)) #sort in the decreasing order
    #trim
    sorted_count_trimmed =  dict(list(sorted_count.items())[:10])

    
    
    
    return sorted_count_trimmed


count=cluster_topic(df_sample, 1)
# {'perfect': 37,
#  'advertis': 1,
#  'shipped': 1,
#  'perfectli': 1,
#  'works': 1,
#  'cake': 1,
#  'topp': 1,
#  's': 1,
#  'condit': 3,
#  'st': 1,
#  'toddl': 1,
#  'thank': 1,
#  'beautiful': 1,
#  'arrived': 1}

ax=plt.subplot(131)
ax.set_title("Cluster"+  str(1))
plt.bar(count.keys(), count.values())
plt.xticks(rotation='vertical')


count=cluster_topic(df_sample, 2)
ax=plt.subplot(132)
ax.set_title("Cluster"+  str(2))
plt.bar(count.keys(), count.values())
plt.xticks(rotation='vertical')


count=cluster_topic(df_sample, 3)
ax=plt.subplot(133)
ax.set_title("Cluster"+  str(3))
plt.bar(count.keys(), count.values())
plt.xticks(rotation='vertical')


count=cluster_topic(df_sample, 4)
ax=plt.subplot(131)
plt.bar(count.keys(), count.values())
ax.set_title("Cluster"+  str(4))

plt.xticks(rotation='vertical')


count=cluster_topic(df_sample, 5)
ax=plt.subplot(132)
ax.set_title("Cluster"+  str(5))
plt.bar(count.keys(), count.values())
plt.xticks(rotation='vertical')


count=cluster_topic(df_sample, 6)
ax=plt.subplot(133)
ax.set_title("Cluster"+  str(6))
plt.bar(count.keys(), count.values())
plt.xticks(rotation='vertical')



count=cluster_topic(df_sample, 7)
ax=plt.subplot(131)
plt.bar(count.keys(), count.values())
ax.set_title("Cluster"+  str(7))

plt.xticks(rotation='vertical')


count=cluster_topic(df_sample, 8)
ax=plt.subplot(132)
ax.set_title("Cluster"+  str(8))
plt.bar(count.keys(), count.values())
plt.xticks(rotation='vertical')


count=cluster_topic(df_sample, 9)
ax=plt.subplot(133)
ax.set_title("Cluster"+  str(9))
plt.bar(count.keys(), count.values())
plt.xticks(rotation='vertical')


count=cluster_topic(df_sample, 0)
ax=plt.subplot(111)
ax.set_title("Cluster"+  str(0))
plt.bar(count.keys(), count.values())
plt.xticks(rotation='vertical')









#------------------------------------------------------------------------------------------------------------------------------------------------
#%% Section 3

#build the word cloud (based on the words in the review)
from wordcloud import WordCloud

#user first 50K for speed
one_bag_words=''.join([word for word in df['review_body'][0:50000]])


wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(one_bag_words)
plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Frequent words used in a sample of 50K reviews", weight='bold', fontsize=20)
plt.show()

#Image saved. Note br br is in the text. The text was NOT pre-processed. This was taw text


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#%% Section 4
#DATA EXPLORATION

#break down by the review dates. Data Format is as follows:
df['review_date'].head()
# 0    2015-08-31
# 1    2015-08-31
# 2    2015-08-31
# 3    2015-08-31
# 4    2015-08-31



df['review_date_YY'] = pd.DatetimeIndex(df['review_date']).year
df['review_date_MM'] = pd.DatetimeIndex(df['review_date']).month
df['review_date_DD'] = pd.DatetimeIndex(df['review_date']).day



df['review_date_YY'].hist(figsize=(15, 9), bins=20, xlabelsize=8, ylabelsize=8);


df['review_date_MM'].hist(figsize=(15, 9), bins=31, xlabelsize=8, ylabelsize=8);


df['review_date_DD'].hist(figsize=(15, 9), bins=31, xlabelsize=8, ylabelsize=8);


#%%
#reviews
plt.bar([5,4,3,2,1], df['star_rating'].value_counts())

plt.xlabel('Ratings')
plt.ylabel('Count')
plt.title('Star Ratings - Binned')


#%%
#word count
df['review_body_WordCount'] = df['review_body'].apply(lambda x: len(x.split())) #count the number of words in each review
df['review_body_WordCount'].max()
# Out[45]: 8859
df['review_body_WordCount'].min()
# Out[46]: 1

df['review_body_WordCount'].hist(figsize=(15, 9), bins=10000,)# xlabelsize=8, ylabelsize=8);


df['review_body_WordCount_log']=np.log(df['review_body_WordCount'])
df['review_body_WordCount_log'].max() #np.log(8859)
# 9.089189170412032
df['review_body_WordCount_log'].min()
#1

df['review_body_WordCount_log'].hist(figsize=(15, 9), bins=20, xlabelsize=8, ylabelsize=8);


df['review_body_WordCount'].describe()
# count    4.859203e+06
# mean     4.786876e+01
# std      7.221935e+01
# min      1.000000e+00
# 25%      1.400000e+01
# 50%      2.800000e+01
# 75%      5.600000e+01
# max      8.859000e+03

df['review_body_WordCount'].value_counts()

# 20      174776
# 2       149349
# 21      145533
# 1       136499
# 22      126151
 
# 2060         1
# 3499         1
# 1430         1
# 1183         1
# 1300         1

df['review_body_WordCount'].value_counts().max()
# 174776
# Max occurence (frequency =174776) is for reviews with 20 words # 20      174776





