1.  A research paper on spam in product reviews


2. I found a pre-trained data set (on wiki data) at https://nlp.stanford.edu/projects/glove/ / referenced at https://github.com/RaRe-Technologies/gensim-data
It does not contain stemmed words.  For example, I checked the 'happi' and the following were "similar" words:
```
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
```

Moreover, I noticed that the 300-dimenstional pre-trained set was providing better synonyms than the 50-dimensional:


```
50-dim:
word2vec_model.similar_by_word('toy', topn=20)
#Using the smaller set
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
```
vs.
```
300-dim
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

```

Conclusion: The 300-dimensional set is more suitable.


For the pre-processing of the text, a different approach was needed. I explored the POS tags of the data.

In a sample of 2K reviews (balanced for 1- / 0- class sentiments), the following POS were most frequent (nouns, adjectives, adverbs, verbs):
```
(POS tag, Count)
# [('NN', 15190),  #NN: noun, common, singular or mass
#  ('JJ', 9746),   #JJ: adjective or numeral, ordinal
#  ('NNS', 4959),  # NNS: noun, common, plural
#  ('RB', 4445),   # RB: adverb
#  ('VBD', 2963),  #VBD: verb, past tense
#  ('VBP', 2370),  #VBP: verb, present tense, not 3rd person singular
#  ('VBG', 1853),   #VBG: verb, present participle or gerund
#  ('VB', 1635),   #VB: verb, base form
#  ('VBN', 1534),   #VBN: verb, past participle
#  ('VBZ', 1208)]   #VBZ: verb, present tense, 3rd person singular

```
![POS in the sample](https://user-images.githubusercontent.com/20401990/154396470-ea4c436a-3fdb-4c4d-b776-52c3d49caf2a.png)


Looking at the difference in the POS distribution in 1- / 0- class sentiments, we see similar ranking, as in the above (NN, JJ, NNS, RB..):

```
Positive class:
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

Negative Class
# [('NN', 8997),   #8997/29171=0.308422748620205
#  ('JJ', 5456), 
#  ('NNS', 2783),
#  ('RB', 2709),  
#  ('VBD', 1912),
#  ('VBP', 1326),
#  ('VBG', 1143),
#  ('VB', 991),
#  ('VBN', 961),
#  ('IN', 671)]

```

![POS_sentiment](https://user-images.githubusercontent.com/20401990/154396469-8bae427e-2d7c-4832-ba2c-291a3499f4c8.png)


With more words in negative reviews, the image is slightly misleading (i.e., the distributions are even more similar than in the image). To help noticing this, I scaled them by the total count of all words in each corresponding set (20513 for class 1, and 29171 for class 0):


![POS_scaled - no sig diff](https://user-images.githubusercontent.com/20401990/154610955-7e0a0acb-f44c-4324-9ecc-5c26eb8ae2fc.png)


Conclusion: there appears to be no significant differences in POS tags in class 0 vs class 1. In other words, based on this sample, people appear  to use similar (distribution-wise) English grammatical structures when complaining or praising products in product reviews.



3. Since the stemming process is not suitable (due to not being included in the set of embeddings), I explored using the process of lemmatization of words in the reviews next.  

```
Original Example: 
"I'm really happy that I bought these posters, they are perfect on my wall and I love the material. The only problem is that they are a bit smaller than what I thought they would be but other than that they are perfect :)"

 Lemmatized Example:
'i really happy that i bought these poster they are perfect on my wall and i love the material the only problem is that they are a bit smaller than what i thought they would be but other than that they are perfect'

```

After lemmatizing the same sample of reviews, I applied a POS "filter" to exclude words by POS tags that likely will not add to the context of the review. For example, such POS tags as interjections or proper nouns. I re-ran the POS analysis on the lemmatized text and sorted the POS tags into two categories, "keep" vs "filer out":

```
#<--- is "Filter out" 
* is "Keep"
#=================================================================
# [('NN', 22605),     * Noun, singular
  # ('DT', 11995),    <--------DT: determiner
#  ('IN', 10600),     <-------- IN: preposition or conjunction, subordinating 
#  ('JJ', 9291),      * JJ: adjective or numeral, ordinal
#  ('RB', 7402),      * RB: adverb
#  ('PRP', 6726),     <--------PRP: pronoun, personal   
#  ('VB', 5277),      * VB: verb, base form
#  ('VBD', 4072),     * VBD: verb, past tense
#  ('VBP', 3299)]     * VBP: verb, present tense, not 3rd person singular
 # ('VBZ', 2909),     * VBZ: verb, present tense, 3rd person singular
 # ('TO', 2516),      <--------TO: "to" as preposition or infinitive marker 
 # ('PRP$', 2072),    <--------PRP$: pronoun, possessive 
 # ('VBN', 1842),     * VBN: verb, past participle
 # ('VBG', 1826),     * VBG: verb, present participle or gerund
 # ('MD', 1531),      <-------- MD: modal auxiliary
 # ('CD', 788),       <--------CD: numeral, cardinal
 # ('RP', 706),       <--------RP: particle
 # ('NNS', 573),      * Plural ! See below. Imperfect lemmatization
 # ('WRB', 517)]      <--------WRB: Wh-adverb
# ('JJR', 462),       * JJR: adjective, comparative
#  ('WDT', 452),      <--------WDT: WH-determiner
#  ('WP', 322),       <--------WP: WH-pronoun
#  ('RBR', 209),      *  RBR: adverb, comparative
#  ('EX', 188),        <--------WEX: existential there
#  (JJS', 176),        *  JJS: adjective, superlative
#  ('PDT', 142),       <-------- PDT: pre-determiner
#  ('FW', 52),         <--------  FW: foreign word
#  ('RBS', 39),        *  RBS: adverb, superlative
#  ('NNP', 9),         <-------- NNP: noun, proper, singular
#  ('UH', 4),          <-------- UH: interjection 
#  ('WP$', 1)]         <--------WP$: WH-pronoun, possessive


````

The POS analysis may not be 100% correct for all words. For example, the following were labeled as NNS (plural nouns); however, not all these words are plural nouns: 

````
# i <--Pronoun
# i <--Pronoun
# people
# people
# behavior <--Singular Noun
# barbies
# i   <--Pronoun
# thanks
# allows   <--Verb
# clothes
# eatos
# mask   <--Singular Noun
# i
# puzzle   <--Singular Noun
# others  <---
# kitty   <--Singular Noun
# christmas
# span   <--Singular Noun
# disappoints
# ours  <---
# kid    <--Singular Noun
``````

Based on the "Keep" filter, I processed the reviews to retain only the select words / select POS tags. The resulting text was as denser summaries of reviews (which was aligned with the intent and my expectations): 

`````
df_sample['review_body'][1]
# "I'm really happy that I bought these posters, they are perfect on my wall and I love the material. The only problem is that they are a bit smaller than what I thought they would be but other than that they are perfect :)"


#after the filtering:
df_sample['review_body_lemmatized_filtered'][1]
#'i really happy i bought poster are perfect wall i love material only problem is are bit smaller i thought be other are perfect'


``````

4. Topic search in the processed text did not produce expected results.

Using the Wiki embeddings,  the 2K sample set and the averaging technique from the last week, the K-means failed to locate more than 1 distinct cluster!

Next, I switched back to the Google News embeddings, increased the sample size to 60K reviews (with a balanced class split).  

Using the two K-means approaches below, I obtained 10 and 5 topic sets; however, these were not insightful (from an inspection of the top 10 words in each topic cluster): 

````
Approach 1:
# km = KMeans( n_clusters=10, init='k-means++',  n_init=100, max_iter=500,random_state=0, algorithm="full")
Approach 2:
# km = KMeans( n_clusters=5, init='k-means++',  n_init=10, max_iter=300,random_state=0, algorithm="auto")
````

Approach 1:
![topic-0](https://user-images.githubusercontent.com/20401990/154614107-49042a5c-bb94-4380-a6e5-297a5d99f058.png)
![4-6](https://user-images.githubusercontent.com/20401990/154614105-69e7d5f0-ac25-44e6-b52c-2089db859506.png)
![7-9](https://user-images.githubusercontent.com/20401990/154614106-8fe1e110-68fc-4d96-9e2d-092496221899.png)

vs. Approach 2:
![Topicv2-1](https://user-images.githubusercontent.com/20401990/154614108-ba9e9934-cd60-4096-a0f5-5b0aeb8a238a.png)
![Topicv2-2](https://user-images.githubusercontent.com/20401990/154614110-06381b6b-26e8-452f-8ef5-20152ccf87ed.png)

I tried to further filter the reviews and leave only singular and plural nouns. Using approach 1, I obtain (again) not insightful topics:


![Nouns1-3](https://user-images.githubusercontent.com/20401990/154614111-7a4eca8a-d524-4bf1-bbbd-6d90d3421c3e.png)
![NOUN-2](https://user-images.githubusercontent.com/20401990/154614112-c2098b74-4517-4481-801d-8f4e27a46ab7.png)
![NOUN3](https://user-images.githubusercontent.com/20401990/154614113-ddca432b-8e51-448f-a957-cf04a5453b0c.png)
![noun-4](https://user-images.githubusercontent.com/20401990/154614115-74e22d21-f152-48fa-9a70-0d5bc754f676.png)










