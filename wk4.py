
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sophie
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk


SampleSize=1000

#%% SECTION 1
# download the Toys Amazon data set
# SKIPPED //  as in the previous week

#%%

df_sample1=df_shuffled[df_shuffled['sentiment']==1][0:SampleSize]
df_sample0=df_shuffled[df_shuffled['sentiment']==0][0:SampleSize]
df_sample=df_sample1.append(df_sample0)

#pre-process (remove stop_words, check if the data is numeric and stem) //  as in the previous week

   
#%%
   
reviews= reviews.set_index('index') #reset the index back
df_sample['review_body_stemmed']=reviews['review_body']

df_sample= df_sample.reset_index()

   
# df_sample['review_body_stemmed'][1]
# Out[22]: 'realli happi bought poster perfect wall love materi problem bit smaller thought would perfect'

# df_sample['review_body'][1]
# Out[23]: "I'm really happy that I bought these posters, they are perfect on my wall and I love the material. The only problem is that they are a bit smaller than what I thought they would be but other than that they are perfect :)"

   
#%%
# use the https://nlp.stanford.edu/projects/glove/ dataset 
import gensim
import numpy as np

file_embeddings='glove-wiki-gigaword-300.txt'
#file_embeddings='love-wiki-gigaword-50.txt'

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(file_embeddings) 

word2vec_model.vector_size
# Out[9]: 300

  
word2vec_model.similar_by_word('toy', topn=20)
#Using the smaller set, we examine the list of similar words and these are not exactly perfect:
 #  [('toys', 0.8808255791664124),
 # ('doll', 0.7762622237205505),
 # ('dolls', 0.7529333829879761),
 # ('shoe', 0.7435762286186218),
 # ('candy', 0.7196512818336487),
 # ('accessories', 0.7080535292625427),
 # ('shop', 0.696685254573822),
 # ('barbie', 0.6843433380126953),
 # ('makers', 0.6824744343757629),
 # ('sells', 0.6747361421585083),
 # ('gadget', 0.6735449433326721),
 # ('jewelry', 0.6697874069213867),
 # ('pet', 0.6689050793647766),
 # ('store', 0.6637514233589172),
 # ('manufacturer', 0.6615467667579651),
 # ('handmade', 0.6575579643249512),
 # ('maker', 0.656119704246521),
 # ('mattel', 0.6557126045227051),
 # ('miniature', 0.6511661410331726),
 # ('brand', 0.649830162525177)]
  
   
#Using the larger (300-dimensional) set the results are a bit better
 #    [('toys', 0.7819609045982361),
 # ('doll', 0.5669493675231934),
 # ('dolls', 0.5529952645301819),
 # ('mattel', 0.5302010774612427),
 # ('hasbro', 0.5201528668403625),
 # ('lego', 0.4896508455276489),
 # ('barbie', 0.47689497470855713),
 # ('manufacturer', 0.4549693167209625),
 # ('candy', 0.45044729113578796),
 # ('robot', 0.4334096908569336),
 # ('pet', 0.4308604300022125),
 # ('accessories', 0.42884621024131775),
 # ('collectible', 0.4226732552051544),
 # ('figurines', 0.42140713334083557),
 # ('animated', 0.4143809974193573),
 # ('retailer', 0.4128083288669586),
 # ('shop', 0.41239139437675476),
 # ('manufacturers', 0.40895313024520874),
 # ('toymaker', 0.4086858630180359),
 # ('makers', 0.40784144401550293)]
   
   
#checking if stemmed  words are associated with similar words that are  semantically close:
word2vec_model.similar_by_word('happi', topn=20)
# Out[34]: 
# [('bayleaf', 0.4716891646385193),
#  ('perplexa', 0.4554847776889801),
#  ('wujing', 0.4534597396850586),
#  ('tongkonan', 0.45094582438468933),
#  ('bëor', 0.45012542605400085),
#  ('süreyya', 0.4496481120586395),
#  ('aubertine', 0.44741368293762207),
#  ('romantico', 0.4466541111469269),
#  ('zety', 0.44330719113349915),
#  ('loebbering', 0.442934513092041),
#  ('hindfeet', 0.44194281101226807),
#  ('dimidiata', 0.4379560947418213),
#  ('amstgeld', 0.4373660087585449),
#  ('ballcaps', 0.43659934401512146),
#  ('mudi', 0.43515899777412415),
#  ('rostratus', 0.4350450038909912),
#  ('monssen', 0.4345400929450989),
#  ('websters', 0.43368998169898987),
#  ('mongkolporn', 0.4333994686603546),
#  ('gumboots', 0.4321073889732361)]

# Based on the manual inspection, unstemmed words produce better results that are closer (semantically).
word2vec_model.similar_by_word('happiness', topn=20)
# [('contentment', 0.6198950409889221),
#  ('joy', 0.6070018410682678),
#  ('satisfaction', 0.542267382144928),
#  ('enjoyment', 0.5416099429130554),
#  ('prosperity', 0.5320949554443359),
#  ('sadness', 0.5278251767158508),
#  ('love', 0.5148369669914246),
#  ('longing', 0.5137263536453247),
#  ('loneliness', 0.5053730607032776),
#  ('well-being', 0.500186562538147),
#  ('despair', 0.47878724336624146),
#  ('pleasure', 0.4786374270915985),
#  ('sense', 0.47482308745384216),
#  ('sorrow', 0.46613022685050964),
#  ('misery', 0.4650653302669525),
#  ('wish', 0.4601428508758545),
#  ('passion', 0.447552889585495),
#  ('desire', 0.4464268982410431),
#  ('realization', 0.44236651062965393),
#  ('yearning', 0.4413985013961792)]
   
   
#%%
import nltk 
nltk.download('averaged_perceptron_tagger')

#Parts of Speech Analysis is described here https://www.nltk.org/book/ch05.html
#I will use the non-stemmed reviews and find out what parts of speech are the most common


reviews= pd.DataFrame()
reviews['review_body']=df_sample['review_body']
reviews= reviews.reset_index()


for i  in range(0, reviews['review_body'].count()):
   tokens= word_tokenize(reviews.loc[i, ('review_body')])
   tokens = [w.lower()  for w in tokens ]
   tokens = [w for w in tokens if not w in stop_words]
   tokens = [w for w in tokens if w.isalpha()]
   reviews.loc[i, ('review_body')]=' '.join(tokens)
    
# reviews= reviews.set_index('index') #reset the index back
df_sample['review_body_NOTstemmed']=reviews['review_body']
# df_sample= df_sample.reset_index()

df_sample['review_body_NOTstemmed'][1]
# Out[71]: 'really happy bought posters perfect wall love material problem bit smaller thought would perfect'

df_sample['review_body'][1]
# Out[72]: "I'm really happy that I bought these posters, they are perfect on my wall and I love the material. The only problem is that they are a bit smaller than what I thought they would be but other than that they are perfect :)"


all_words=' '.join([word for word in df_sample['review_body_NOTstemmed']]) #join with " " to ensure the adjacent reviews' end and start are spearate



all_words_tokenized = word_tokenize(all_words)
out=nltk.pos_tag(all_words_tokenized)
tag_fd = nltk.FreqDist(tag for (word, tag) in out)


tag_plot=tag_fd.most_common(10)
# [('NN', 15190),   #NN: noun, common, singular or mass
#  ('JJ', 9746),    #JJ: adjective or numeral, ordinal
#  ('NNS', 4959),   # NNS: noun, common, plural
#  ('RB', 4445),    # RB: adverb
#  ('VBD', 2963),   #VBD: verb, past tense
#  ('VBP', 2370),   #VBP: verb, present tense, not 3rd person singular
#  ('VBG', 1853),   #VBG: verb, present participle or gerund
#  ('VB', 1635),    #VB: verb, base form
#  ('VBN', 1534),   #VBN: verb, past participle
#  ('VBZ', 1208)]   #VBZ: verb, present tense, 3rd person singular


#to get the definitions of the tags, use the following:
nltk.download('tagsets')
nltk.help.upenn_tagset('VB')


import matplotlib.pyplot as plt
plt.bar(tag_fd.keys(), tag_fd.values())


#Is the distribution different from positive vs negative sentiment date?
all_words=''
for i  in range(0, df_sample['review_body'].count()):
    if df_sample['sentiment'][i]==1:
        all_words+=df_sample['review_body_NOTstemmed'][i]+' ' #join with " " to ensure the adjacent reviews' end and start are spearate

all_words_tokenized = word_tokenize(all_words)
out=nltk.pos_tag(all_words_tokenized)
tag_fd = nltk.FreqDist(tag for (word, tag) in out)  #freq

length_all_words_tokenized=(len(all_words_tokenized))
# 20513

tag_fd_scaled=[k/length_all_words_tokenized  for  k in tag_fd.values()]


tag_fd.most_common(10)
# [('NN', 6193),    #6193/20513=0.3019061083215522
#  ('JJ', 4290),
#  ('NNS', 2176),
#  ('RB', 1736),
#  ('VBD', 1051),
#  ('VBP', 1044),
#  ('VBG', 710),
#  ('VB', 644),
#  ('VBZ', 591),
#  ('VBN', 573)]


#Proceeding to the Negative sentiment:
all_words=''
for i  in range(0, df_sample['review_body'].count()):
    if df_sample['sentiment'][i]==0:
        all_words+=df_sample['review_body_NOTstemmed'][i]+' ' #join with " " to ensure the adjacent reviews' end and start are spearate

all_words_tokenized = word_tokenize(all_words)
out0=nltk.pos_tag(all_words_tokenized)
tag_fd0 = nltk.FreqDist(tag for (word, tag) in out0)

all_words[0:40]
# total waste money game boring people pra'

length_all_words1_tokenized=(len(all_words_tokenized))
# 29171

tag_fd_scaled0=[k/length_all_words1_tokenized  for  k in tag_fd0.values()]

tag_fd0.most_common(10)
# [('NN', 8997),  #more nouns    8997/29171=0.308422748620205
#  ('JJ', 5456),  #more adjectives
#  ('NNS', 2783), #noun plural
#  ('RB', 2709),  #adverb
#  ('VBD', 1912),
#  ('VBP', 1326),
#  ('VBG', 1143),
#  ('VB', 991),
#  ('VBN', 961),
#  ('IN', 671)]



plt.subplot(121)
plt.bar(tag_fd.keys(), tag_fd.values())
plt.xticks(rotation='vertical')
plt.subplot(122)

plt.bar(tag_fd0.keys(), tag_fd0.values())
plt.suptitle('POS for Positive (left) vs. Negative (right) Sentiments ')
plt.xticks(rotation='vertical')



#plot the scaled POS freq. in the sample
plt.subplot(121)
plt.bar(tag_fd.keys(), tag_fd_scaled)
plt.xticks(rotation='vertical')
plt.subplot(122)

plt.bar(tag_fd0.keys(), tag_fd_scaled0)
plt.suptitle('POS for Positive (left) vs. Negative (right) Sentiments ')
plt.xticks(rotation='vertical')

#Note: stemming is not be acceptable; create a filter based on the POS 



#%%

#TRY LEMMATIZAZATION instead of stemming
nltk.download('wordnet')

from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()


reviews= pd.DataFrame()
reviews['review_body']=df_sample['review_body']
reviews= reviews.reset_index()


for i  in range(0, reviews['review_body'].count()):
   tokens= word_tokenize(reviews.loc[i, ('review_body')])
   tokens = [w.lower()  for w in tokens ]
   # tokens = [w for w in tokens if not w in stop_words]   Removed  because it removes the "not" in reviews
   tokens = [w for w in tokens if w.isalpha()]
   tokens = [lemma.lemmatize(w) for w in tokens]
   reviews.loc[i, ('lemmatized')]=' '.join(tokens)

   
# reviews= reviews.set_index('index') #reset the index back
df_sample['review_body_lemmatized']=reviews['lemmatized']

# df_sample= df_sample.reset_index()
df_sample['review_body_NOTstemmed'][1]
# Out[71]: 'really happy bought posters perfect wall love material problem bit smaller thought would perfect'


df_sample['review_body'][1]
# Out[72]: "I'm really happy that I bought these posters, they are perfect on my wall and I love the material. The only problem is that they are a bit smaller than what I thought they would be but other than that they are perfect :)"

df_sample['review_body_lemmatized'][1]
 # 'i really happy that i bought these poster they are perfect on my wall and i love the material the only problem is that they are a bit smaller than what i thought they would be but other than that they are perfect'


#POS Freq Distributions for lemmatized reviews
all_words=' '.join([word for word in df_sample['review_body_lemmatized']]) 
all_words_tokenized = word_tokenize(all_words)
out=nltk.pos_tag(all_words_tokenized)
tag_fd = nltk.FreqDist(tag for (word, tag) in out)


tag_plot=tag_fd.most_common(32)   #32 in total
#  <--- is FILTER OUT and * is KEEP
# [('NN', 22605),     *KEEP
  # ('DT', 11995),    <--------DT: determiner
#  ('IN', 10600),    <-------- IN: preposition or conjunction, subordinating 
#  ('JJ', 9291),      *KEEP JJ: adjective or numeral, ordinal
#  ('RB', 7402),       *KEEP RB: adverb
#  ('PRP', 6726),     <--------PRP: pronoun, personal   
#  ('VB', 5277),      *KEEP  VB: verb, base form
#  ('VBD', 4072),      *KEEP VBD: verb, past tense
#  ('VBP', 3299)]     *KEEP  VBP: verb, present tense, not 3rd person singular
 # ('VBZ', 2909),      *KEEP VBZ: verb, present tense, 3rd person singular
 # ('TO', 2516),      <--------TO: "to" as preposition or infinitive marker 
 # ('PRP$', 2072),    <--------PRP$: pronoun, possessive 
 # ('VBN', 1842),     *KEEP VBN: verb, past participle
 # ('VBG', 1826),      *KEEP VBG: verb, present participle or gerund
 # ('MD', 1531),      <-------- MD: modal auxiliary
 # ('CD', 788),       <--------CD: numeral, cardinal
 # ('RP', 706),       RP: particle<--------
 # ('NNS', 573),       *KEEP. Plural ! See below. Imperfect lemmatization
 # ('WRB', 517)]      <--------WRB: Wh-adverb
# ('JJR', 462),        *KEEP JJR: adjective, comparative
#  ('WDT', 452),       <--------WDT: WH-determiner
#  ('WP', 322),       <--------WP: WH-pronoun
#  ('RBR', 209),        *KEEP    RBR: adverb, comparative
#  ('EX', 188),        <--------WEX: existential there
#  (JJS', 176),        *KEEP  JJS: adjective, superlative
#  ('PDT', 142),        <---  PDT: pre-determiner
#  ('FW', 52),          <---  FW: foreign word
#  ('RBS', 39),        *KEEP  RBS: adverb, superlative
#  ('NNP', 9),          <--- NNP: noun, proper, singular
#  ('UH', 4),          <--- UH: interjectionUH: interjection
#  ('WP$', 1)]        <---WP$: WH-pronoun, possessive

nltk.help.upenn_tagset('VBD')

#find the words that are plural nouns:
for i in range (0, len(out )):
    if out[i][1]=='UH': 
        print (out[i][0])      
# oh
# oh
# oh
# yes
    
#print the first 20 plural nouns    
count=0
for i in range (0, len(out )):
    if out[i][1]=='NNS' and count <=20: 
        print (out[i][0])
        count+=1
#these are not  all plural nounces        
# i
# i
# people
# people
# behavior <<
# barbies
# i    <<
# thanks
# allows   <<
# clothes
# eatos
# mask
# i
# puzzle
# others
# kitty
# christmas
# span
# disappoints
# ours
# kid
   

#%%

#Filter out and keep  review words the following POS tags: 
# [('NN', 22605),     *KEEP
#  ('JJ', 9291),      *KEEP JJ: adjective or numeral, ordinal
#  ('RB', 7402),       *KEEP RB: adverb
#  ('VB', 5277),      *KEEP  VB: verb, base form
#  ('VBD', 4072),      *KEEP VBD: verb, past tense
#  ('VBP', 3299)]     *KEEP  VBP: verb, present tense, not 3rd person singular
 # ('VBZ', 2909),      *KEEP VBZ: verb, present tense, 3rd person singular
 # ('VBN', 1842),     *KEEP VBN: verb, past participle
 # ('VBG', 1826),      *KEEP VBG: verb, present participle or gerund
 # ('NNS', 573),        <----plural nouns! See below. Imperfect lemmatization
# ('JJR', 462),        *KEEP JJR: adjective, comparative
#  ('RBR', 209),        *KEEP    RBR: adverb, comparative
#  (JJS', 176),        *KEEP  JJS: adjective, superlative
#  ('RBS', 39),        *KEEP  RBS: adverb, superlative


POS_filer={'JJ', 'NN', 'RB', 'VB', 'VBD', 'VBP', 'VBZ', 'VBN', 'VBG', 'NNS', 'JJR','RBR', 'JJS', 'RBS' } 


df_sample['review_body_lemmatized_filtered'] = df_sample['review_body_lemmatized'] 


for i in range (0, df_sample.shape[0]):
    filtered_out=''
    one_review=df_sample['review_body_lemmatized'][i]    
    tokenized = word_tokenize(one_review)
    out=nltk.pos_tag(tokenized)
    filtered_out = ' '.join ([word for  word,k in out if  k in POS_filer] )
    df_sample['review_body_lemmatized_filtered'][i]=filtered_out


df_sample['review_body'][1]
 # "I'm really happy that I bought these posters, they are perfect on my wall and I love the material. The only problem is that they are a bit smaller than what I thought they would be but other than that they are perfect :)"


#the review after the filtering:
df_sample['review_body_lemmatized_filtered'][1]
#'i really happy i bought poster are perfect wall i love material only problem is are bit smaller i thought be other are perfect'




#%%
#Find the topics in lemmatized and filtered reviews


#check if the words are in the word2vc vocabulary 
df_sample['review_body_lemmatized_filteredw2v']=df_sample['review_body_lemmatized_filtered']


for i  in range(0, df_sample['review_body_lemmatized_filtered'].count()):
   tokens= word_tokenize(df_sample.loc[i, ('review_body_lemmatized_filtered')])
   tokens = [w for w in tokens if w in word2vec_model.key_to_index ]
   df_sample.loc[i, ('review_body_lemmatized_filteredw2v')]=' '.join(tokens)



#using the 300-dim embedding from the stanford
#representing each review by the mean of word embeddings for the words used in the review body.
word2vec_model_embeddings = WordVecVectorizer(word2vec_model)
word2vec_model_embeddings_ave = word2vec_model_embeddings.transform(df_sample['review_body_lemmatized_filteredw2v'])
print(word2vec_model_embeddings_ave.shape)
# (2000, 300)



from sklearn.cluster import KMeans
km = KMeans( n_clusters=10, init='k-means++',  n_init=100, max_iter=500,random_state=0, algorithm="full")
y_km = km.fit_predict(word2vec_model_embeddings_ave)
df_sample['Topic_Cluster'] = pd.DataFrame(y_km)
#%%
# ConvergenceWarning: Number of distinct clusters (1) found smaller than n_clusters (10). Possibly due to duplicate points in X.



km = KMeans( n_clusters=5, init='k-means++',  n_init=10, max_iter=300,random_state=0, algorithm="auto")
y_km = km.fit_predict(word2vec_model_embeddings_ave)
df_sample['Topic_Cluster'] = pd.DataFrame(y_km)
# /Users/sophie/opt/anaconda3/lib/python3.8/site-packages/sklearn/cluster/_kmeans.py:1077: ConvergenceWarning: Number of distinct clusters (1) found smaller than n_clusters (5). Possibly due to duplicate points in X.



#%%
##Taking A bigger sample

SampleSize=40000


df_sample1=df_shuffled[df_shuffled['sentiment']==1][0:SampleSize]
df_sample0=df_shuffled[df_shuffled['sentiment']==0][0:SampleSize]
df_sample=df_sample1.append(df_sample0)



df_sample['review_body'][176]
# "My son and I LOVE this toy. In fact, it is such a nice toy that many of my friends have purchased this for their children after visiting us. The only complaint that I have is that the case doesn't latch shut and everything falls ou the moment you move it. I had to purchse a plastic bin to keep it in."

df_sample['review_body_lemmatized'][176]
#  'my son and i love this toy in fact it is such a nice toy that many of my friend have purchased this for their child after visiting u the only complaint that i have is that the case doe latch shut and everything fall ou the moment you move it i had to purchse a plastic bin to keep it in'

df_sample['review_body_lemmatized_filtered'][176]
# 'son i love toy fact is such nice toy many friend have purchased child visiting only complaint i have is case doe latch shut everything fall ou moment move i had purchse plastic bin keep'

df_sample['review_body_lemmatized_filteredw2v'][176]
# 'son i love toy fact is such nice toy many friend have purchased child visiting only complaint i have is case doe latch shut everything fall ou moment move i had plastic bin keep'
#purchse fell out due to not being the data set of embeddings



#Go back to the the Google News embeddings:
# print(word2vec_model_embeddings_ave.shape)

# Output from spyder call 'get_namespace_view':
# (80000, 300)
# Use the kmeans
# km = KMeans( n_clusters=10, init='k-means++',  n_init=100, max_iter=500,random_state=0, algorithm="full")



# df_sample['Topic_Cluster'].value_counts()
# Out[437]: 
# 4    24691
# 3    20637
# 8    16398
# 1    11251
# 0     3114
# 5     1220
# 6     1022
# 2      891
# 7      578
# 9      198


#%%
def cluster_topic(df_sample, topic_number=1):
    df_filtered = df_sample.pipe(lambda x: x.loc[x['Topic_Cluster'] == topic_number])
    #stem and the words 
    list_reviews= ( [ ps.stem(word) for word in df_filtered['review_body_lemmatized_filteredw2v'] ])
    
    #count freq of words in the reviews in each cluester
    from collections import Counter
    count = dict(Counter(word for sentence in list_reviews for word in sentence.split()))
    
    sorted_count = dict( sorted(count.items(),   key=lambda item: item[1], reverse=True)) #sort in the decreasing order
    #trim to show 10
    sorted_count_trimmed =  dict(list(sorted_count.items())[:10])
    return sorted_count_trimmed

#%%

count=cluster_topic(df_sample, 1)
# {'i': 5336,
#  'is': 5200,
#  'love': 5109,
#  'old': 3488,
#  'year': 2886,
#  'wa': 2762,
#  'great': 2619,
#  'toy': 2544,
#  'daughter': 2449,
#  'son': 1926}


#Plotting / Skipped

# km = KMeans( n_clusters=5, init='k-means++',  n_init=10, max_iter=300,random_state=0, algorithm="auto")
#3    43239
# 2    20239
# 0    12590
# 4     2332
# 1     1600
#Plotting / Skipped


#%% Restricting to Nouns only
POS_filer={'NN', 'NNS' } 
df_sample['review_body'][1]
# Out[467]: 'This is a great craft/building project.  Low mess and easy for kids to create their own masterpieces with minimal help.  I ordered several of these and filled clear cellophane gift bags with them for a great birthday party favor.'

df_sample['review_body_lemmatized_filtered'][1]
# 'project mess kid masterpiece help cellophane gift bag birthday party favor'

# km = KMeans( n_clusters=10, init='k-means++',  n_init=100, max_iter=500,random_state=0, algorithm="full")


df_sample['Topic_Cluster'].value_counts()
# 0    28737
# 4    21807
# 3    10555
# 7     4732
# 9     3768
# 2     3023
# 5     2821
# 6     2762
# 1      903
# 8      892


count=cluster_topic(df_sample, 1)

# {'gift': 846,
#  'thank': 169,
#  'christmas': 119,
#  'grandson': 59,
#  'i': 57,
#  'year': 52,
#  'son': 37,
#  'birthday': 36,
#  'wa': 32,
#  'item': 30}




