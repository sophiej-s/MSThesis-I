# Approach #2 / Design #2

Using the "Toys" dataset, perform the topic modeling for groups of data (ratings 1-5). One topic is found per each rating collection. 
Perform reduction / unique processing and also obtain embeddings. Perform rating prediction as well as sentiment prediction.

![slide-image 001](https://user-images.githubusercontent.com/20401990/161186348-11c1b339-d34a-443b-b4bf-18c9259043f7.jpeg)


The resulting topics are reduced to 3D and expectedly do not overlap in the direcitons:

![APPROACH 2](https://user-images.githubusercontent.com/20401990/161186375-9101d3ba-e2bf-4b3b-9ff9-960beb059642.png)



For ratings 1-5:  Calculate the distances and use the SVM to predict the ratings. The performance is quite poor. 
Sample: 1K per each rating = 5K samples
````
Features: 5K x 315
LinearSVC

#               precision    recall  f1-score   support

#            1   0.441176  0.560166  0.493601       241
#            2   0.324873  0.266667  0.292906       240
#            3   0.288991  0.258197  0.272727       244
#            4   0.346734  0.255556  0.294243       270
#            5   0.460606  0.596078  0.519658       255

#     accuracy                       0.386400      1250
#    macro avg   0.372476  0.387333  0.374627      1250
# weighted avg   0.372704  0.386400  0.374207      1250

````

SVM with the rbf kernel did not perform much better:
````
#               precision    recall  f1-score   support

#            1   0.453488  0.485477  0.468938       241
#            2   0.310924  0.308333  0.309623       240
#            3   0.291304  0.274590  0.282700       244
#            4   0.414286  0.322222  0.362500       270
#            5   0.528662  0.650980  0.583480       255

#     accuracy                       0.408800      1250
#    macro avg   0.399733  0.408321  0.401448      1250
# weighted avg   0.401325  0.408800  0.402372      1250
````

Define the sentiment. Reduce the dataset by removing 3's and observe improved statistics.
Sample: 2K per each sentiment = 4K samples

````
#               precision    recall  f1-score   support

#            0   0.773063  0.838000  0.804223       500
#            1   0.823144  0.754000  0.787056       500

#     accuracy                       0.796000      1000
#    macro avg   0.798103  0.796000  0.795640      1000
# weighted avg   0.798103  0.796000  0.795640      1000


````





