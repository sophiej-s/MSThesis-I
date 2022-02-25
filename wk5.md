## Topic Modeling using Tf-idf and Latent Dirichlet Allocation 


A small sample (2K reviews in total) was pre-processed by removing the strings indicating breaks  (```<br>```) and numeric characters, lemmatizing (NLTK WordNetLemmatizer) and stemming (NLTK PorterStemmer).


The TF-IDF method was applied (with a minimum limit for occurrences of words set to be 10 reviews) to the data followed by the sklearn's LDA (the learning method was set to 'batch') to model the topics.
Five topics (top 50 words per each) were identified on the whole set as follows:

````
# Topic0
# love old year toy play daughter gift doll cute grandson bought fun littl like son kid great granddaught small christma realli got child month enjoy babi girl nephew time lot nice niec birthday set size boy purchas easi soft good togeth perfect want ok yr favorit use age make grand

# Topic1
# great game love perfect thank product card excel item good parti arriv son happi ship fast fun kid exactli time play condit price describ qualiti costum recommend look work use came order collect lego set like gift purchas want receiv size littl got expect nice deliveri year halloween beauti realli

# Topic2
# good expect look nice figur smaller awesom price qualiti pictur cool like color small realli great thought product size love amaz work fit cute toy mask expens paint littl cheapli pretti bigger better costum ador collect job cheap transform sticker disappoint way wear daughter buy use kid differ set dress

# Topic3
# work batteri money buy game product return time like day got wast disappoint broke use order thing toy fli bought worth tri car item receiv charg amazon play cheap new realli fun control son review replac box broken light open look purchas minut junk make week came christma price great

# Topic4
# piec use puzzl like plastic togeth look qualiti toy make box time play set product good littl work disappoint come realli kid cheap great game easili son figur apart tri fun fall hard hold thing child way nice broke need love stay fit model kit purchas buy better howev came

````
Given the data set is for reviews of toys, possible topic-level identifiers based on the above data are as follows: 

* Topic 0 is probably centered on reviews of toys for smaller children (possibly gifted by older relatives) due to the following words: ````babi, child, year, old, granddaught, grandson, grand, niece, age````

* Topic 1 can be summarized as revies of shipping (````ship, receive, deliveri````) and notes as to the good qualities of the products

* Topic 2 is probably for toy costumes and reviews of the fit of those based on the following: ````smaller, small, size, fit, mask, bigger, costum, wear, dress````

* Topic 3 is probably for reviews that complain about returns (````return, disappoint````) due to product deficiencies (````cheap, broke, broken, junk````)

* Topic 4 is probably for puzzles (or possibly more broadly for board games) and reviews noting good quality (````good, easili, fun, nice````) and an affordable pice (````cheap````)

Performing a TSNE transformation to 3D (from 5D) and mapping the data reveals spatial separation of the topics.
![FIG1](https://user-images.githubusercontent.com/20401990/155641352-0cc656aa-8a1a-4dfa-bb3e-7d9cff86d1a0.png)



Using the same data, I also examined the topics in reviews with negative (star rating of 4 or below) and positive (star rating of 5 only):

Positive Sentiment:
```
# Topic0
# figur awesom great work excel like look set good use need realli model collect price nice qualiti amaz fli come buy love product make littl box batteri thing kit recommend ok better game play cool piec time card paint fan servic star best toy pretti build plastic high addit purchas

# Topic1 
# love great gift grandson granddaught daughter nice son old year christma kid bought toy happi birthday littl niec expect like got perfect grand price qualiti ador nephew purchas present doll cute boy realli work play size friend good look girl item set product enjoy babi absolut parti thank want wait

# Topic2
# game thank great product perfect card ship fast arriv time good condit play describ item exactli fun love deliveri lego famili quick came quickli excel enjoy buy want easi expect wonder order purchas nice recommend look use qualiti price pleas son packag friend alway like kid happi make seller receiv

# Topic3 
# good fun cute game play love great old toy kid year lot realli child son use recommend learn cool super easi qualiti like gift littl bought soft beauti set togeth parti time enjoy costum grandson hour color daughter highli educ big adult famili durabl hit fan perfect help got look

# Topic4 
# love doll toy daughter old like great littl year play puzzl bought use fit look son got month realli piec small make kid cute time girl babi togeth thing good size perfect easi work worth day fun want set nice buy child lot come sturdi think pretti order qualiti purchas


```

*  Topic 0 contains reviews on price and appearance of toys
*  Topic 1 contains reviews centered on family gift-giving
*  Topic 2 contains reviews of shipping or delivery of toys
*  Topic 3 contains reviews of toys for learning and education 
*  Topic 4 contains reviews on product quality

A plot of the TSNE 3D-transformed data reveals the data is separable:
![FIGURE2-positive sentiment only](https://user-images.githubusercontent.com/20401990/155641351-030fb72b-b1fe-4e92-8f51-439dcdfc444f.png)


Similarly, examining reviews with the negative sentiment, one can identify the topics and note the TSNE 3D-transformed data also appears separable:

````
# Topic0 
# work toy game play like time make old son batteri year use realli fun bought kid tri thing love child car set great littl got turn buy sound product return want review good think daughter purchas look christma disappoint hard money better way month day track need new anoth card

# Topic1 
# qualiti batteri product poor paint fli charg light buy cheap bad use money helicopt work expens good control figur plane box blade time terribl wast horribl look job need plastic like glow receiv item color disappoint wood great red order fragil piec return thing broke pop realli come model purchas

# Topic2 
# broke money small cheap smaller worth cheapli expect toy price cute thought work day love wast old son kid disappoint play grandson got realli year junk bought like littl use buy christma fun way good flimsi open daughter recommend okay item paid pretti gift cost plastic car return qualiti bigger

# Topic3 
# small order receiv box product pictur ok puzzl broken arriv card item came miss like packag gift piec look disappoint ship good return got balloon open pack advertis expect buy use happi size descript great seller purchas damag say price bought work sent bag took want amazon send plastic set

# Topic4 
# like look use good piec size daughter doll fit disappoint expect small apart littl old pictur love cute qualiti year togeth cheap toy nice kid fall bought plastic realli ball easili color product child son time play hold come long price thought hard stay better head great fell costum way

`````
*  Topic 0 contains reviews in which issues or defects, that lead to returns of the products, are discussed
*  Topic 1 contains reviews where customers noted their disappointment in prodct quality
*  Topic 2 contains reviews where customers noted disappointment in the quality per price 
*  Topic 3 contains reviews where customers discuss issues with shipping 
*  Topic 4 contains reviews where customers discuss being disappointed in in the look or appearance of products

![FIG3-NEGATIVE ONLY](https://user-images.githubusercontent.com/20401990/155641350-0e775c9d-d0d1-4b8b-8f86-47e479bc07f1.png)


Taking a larger sub-set of the data (60K per each sentiment), we observe the TSNE transformations are a little more difficult to interpret as separable (based on a visual inspection):
![TSNE-60K-2](https://user-images.githubusercontent.com/20401990/155641347-f9ea91b3-9d72-4e26-9965-189b5f0a290a.png)
![TSNE-60KEACH](https://user-images.githubusercontent.com/20401990/155641349-361c2df7-4b03-461a-81b2-80cfc5c34bb4.png)

The topic modeling using the learger data set is less obvious compared to examining a smaller set of only one sentiment at a time.

````
# Topic0
# great product good price item card arriv qualiti ship order excel buy money small expect box work came receiv disappoint return ok look worth purchas happi describ fast gift cheap time packag thank like amazon got toy condit love broken exactli want seller thought bought pictur wast son store christma

# Topic1
# love game play old fun year great kid toy gift daughter son grandson bought like littl granddaught christma enjoy child time realli got birthday use lot set friend cute awesom famili good make perfect month recommend easi card age learn nephew think purchas babi want boy thing niec girl hit

# Topic2
# work batteri car toy use son time set track fli train great like love old make light control year charg bought play fun turn realli truck tri got kid wheel buy thing need good littl product lego disappoint stop easi sound helicopt replac togeth minut motor build run grandson money

# Topic3
# doll cute look like nice love costum figur expect perfect size fit color littl great daughter pictur small qualiti good realli dress wear beauti thank collect head hat soft ador hair smaller big girl cool halloween pretti paint price year old bought barbi super better got mask face want fan

# Topic4
# piec good use puzzl broke togeth like plastic toy apart kid easili look time fall water day qualiti product realli littl play come work hold great stay hard easi love make old ball cheap son nice disappoint buy tri box year small break thing money got hole open set fell




````


