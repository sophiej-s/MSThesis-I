# Week 3 Summary



# Section 1
I re-ran the previous week's analysis (except skipping the sorting by the "most helpful") to determine the impact of stemming and the training size on the accuracy. Accuracy went up with 600K reviews and stemming also positively influenced the results. 

  

2K reviews (total with 1K data points for class 0 and 1K for class 1) not sorted by the most helpful and stemmed: 

```
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.72      0.68      0.70       302
#       1 = positive (rating 4 or 5)       0.70      0.73      0.71       298

#                           accuracy                           0.71       600
#                          macro avg       0.71      0.71      0.71       600
#                       weighted avg       0.71      0.71      0.71       600

```
vs. 
2K  reviews (total) not sorted by the most helpful and NOT stemmed:
```
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.71      0.70      0.70       309
#       1 = positive (rating 4 or 5)       0.68      0.70      0.69       291

#                           accuracy                           0.70       600
#                          macro avg       0.70      0.70      0.70       600
#                       weighted avg       0.70      0.70      0.70       600

```

When data were not non-stemmed, we observe lower accuracy and precision but higher recall (for class 0). 

Next, we select a bigger data sample (20K total reviews). Below are the results for stemmed data 

```
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.83      0.79      0.81      3043
#       1 = positive (rating 4 or 5)       0.79      0.83      0.81      2957

#                           accuracy                           0.81      6000
#                          macro avg       0.81      0.81      0.81      6000
#                       weighted avg       0.81      0.81      0.81      6000
```
vs

The same 20K reviews--not stemmed

```
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.81      0.80      0.81      2993
#       1 = positive (rating 4 or 5)       0.81      0.81      0.81      3007

#                           accuracy                           0.81      6000
#                          macro avg       0.81      0.81      0.81      6000
#                       weighted avg       0.81      0.81      0.81      6000
```
The accuracy improved between the stemmed vs non-stemmed.



Increasing the data sample to 60K (total) again leads to results that are close (stemmed vs un-stemmed, below).
```
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.85      0.82      0.83      9028
#       1 = positive (rating 4 or 5)       0.83      0.85      0.84      8972

#                           accuracy                           0.84     18000
#                          macro avg       0.84      0.84      0.84     18000
#                       weighted avg       0.84      0.84      0.84     18000

```

vs.
```
# 0 = negative (rating lower than 3)       0.84      0.82      0.83      9072
#       1 = positive (rating 4 or 5)       0.82      0.85      0.84      8928

#                           accuracy                           0.83     18000
#                          macro avg       0.83      0.83      0.83     18000
#                       weighted avg       0.83      0.83      0.83     18000

```

Similarly, the benefit of stemming is not easy to call out when the data is organized by the "most helpful" and the N number of most helpful reviews are sampled (stemmed vs un-stemmed, below)

```
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.84      0.82      0.83      8986
#       1 = positive (rating 4 or 5)       0.82      0.84      0.83      9014

#                           accuracy                           0.83     18000
#                          macro avg       0.83      0.83      0.83     18000
#                       weighted avg       0.83      0.83      0.83     18000
```
vs. 
```
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.84      0.83      0.83      9018
#       1 = positive (rating 4 or 5)       0.83      0.84      0.84      8982

#                           accuracy                           0.84     18000
#                          macro avg       0.84      0.84      0.84     18000
#                       weighted avg       0.84      0.84      0.84     18000
```


Increasing the size drastically leads to increased accuracy, precision, f1. The results below are for 600K (total), stemmed reviews organized by the "most helpful":

```
# [nltk_data]   Package stopwords is already up-to-date!
#                                     precision    recall  f1-score   support

# 0 = negative (rating lower than 3)       0.86      0.86      0.86     89702
#       1 = positive (rating 4 or 5)       0.86      0.86      0.86     90298

#                           accuracy                           0.86    180000
#                          macro avg       0.86      0.86      0.86    180000
#                       weighted avg       0.86      0.86      0.86    180000

```





# Section 2. Analysis of topics (clusters) in the reviews using word2vec.
Corpus of trained data is at https://code.google.com/archive/p/word2vec/
I used the existing pre-trained corpus for News (GoogleNews-vectors-negative300.bin.gz). The reason was the data was readily available (no need to train) and there was no reviews-centered corpus readily available.

Words are represented with 300-dimenstional features and the corpus contains stemmed words.  
```
# 'beauti littl set granddaught love'
#  'beauti' in word2vec_model.key_to_index
# Out[78]: True

```

Taking a small sample of reviews (9718), we represent each review by the mean of word embeddings for the words used in the review body and then use the K-Means to locate 10 (note that 10 was selected arbitrarily after experimenting with 8 and 50) clusters in the reviews.

To depict the results, I used the TSNE method to reduce the 300 dimensions to 3, thus changing the size of embeddings from (97184, 300) to (9718, 3).
When the 10 clusters are colored with distinct colors, one can notice the (at least) one distinct cluster (purple, two Figures below)



![Cluster3d-color2](https://user-images.githubusercontent.com/20401990/153539099-c71ed108-0f6b-4d92-95f4-bafd69c45b5c.png)
![Cluster3d-color](https://user-images.githubusercontent.com/20401990/153539101-23593835-517e-476d-8280-5394b07a7934.png)


In terms of the number of reviews that belong to each cluster, clusters 0 contains the most reviews and cluster 1 contains the least: 

```
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

```
Looking at cluster 1 more closely we see it contains mostly reviews containing the word "perfect":
<img width="491" alt="Screen Shot 2022-02-10 at 9 43 23 PM" src="https://user-images.githubusercontent.com/20401990/153539301-7c76d078-069c-4916-a935-d990eee22bf3.png">


Examining the 10 most frequent words in each topic cluster:

![Clusetr0](https://user-images.githubusercontent.com/20401990/153539420-c59fc7ba-98dc-4319-af54-b7f3015855e6.png)
Cluster 0's most frequent word is "br" (likely not the result of stemming a legitimate word). We need to review the pre-processing steps. The rest of the words are not providing hints to any specific domain/topic


![Clusters1-3](https://user-images.githubusercontent.com/20401990/153539423-109ae20e-fc36-4fb7-9dc0-b3182e1e09da.png)

Cluster 1 was discussed above. Clusters 2-3 consist of positive words that are mostly adjectives.


![Cluster4-6](https://user-images.githubusercontent.com/20401990/153539422-ffb5cb7e-4f41-4d42-a0af-9196bfa32495.png)

Clusters 4-6 have the "love" in common, cluster 5 possibly is on a topic of family (due to "son", "granddaughter", "grandson").

![Clusters7-9](https://user-images.githubusercontent.com/20401990/153539421-fe368903-2093-4b36-a0fd-b83cf845b2f4.png)

Clusters 7-9 also contain positive words like love, perfect, great. Perhaps only cluster 1 stands out as special among these clusters. 


# Section 3. Word Cloud 
Note the prominence of br br
![WordCloud](https://user-images.githubusercontent.com/20401990/153540123-32361546-e397-47cd-8f33-319293c98093.png)


# Section 4. Data exploration

Review counts by Month, Year, Day. 

![Review_Countby_Year](https://user-images.githubusercontent.com/20401990/153540268-1c1cfaef-b1b4-4cf1-8097-1ed49184fae9.png)
There is a sharp increase in the number of reviews after 2012.


![Review_Countby_Day](https://user-images.githubusercontent.com/20401990/153540267-a39b36cd-637c-44fe-8190-dc6a5acb3043.png)
January and December are the most productive months, yielding more reviews than in the other months. 

![Review_Count_ByMonth](https://user-images.githubusercontent.com/20401990/153540269-4acecd3e-b4a6-406a-b27b-2377fd4b3ee1.png)
The number of reviews written on day 31 drops (which can be explained that not all months have 31 days). 




Examining the "review body" column, we see the mean number of words in a review is 478, minimum is 1 (i.e., "Perfect") and maximum is 8.8K words in a review!
```
df['review_body_WordCount'].describe()
# count    4.859203e+06
# mean     4.786876e+01
# std      7.221935e+01
# min      1.000000e+00
# 25%      1.400000e+01
# 50%      2.800000e+01
# 75%      5.600000e+01
# max      8.859000e+03

```


Most frequent (frequency of 174776) are  reviews with 20 words:
```
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


```

![Star-Ratings_binned](https://user-images.githubusercontent.com/20401990/153540643-0104a5f4-b251-4cca-b171-4eb02d8c4235.png)
![Word_Count_Log](https://user-images.githubusercontent.com/20401990/153540644-e12827e7-69e9-4d31-bfb4-a7ebd11ef59f.png)
![Word_Count](https://user-images.githubusercontent.com/20401990/153540645-b04d7a05-bc62-4402-b367-3dc765679361.png)






