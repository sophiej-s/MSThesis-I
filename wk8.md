# MSThesis-I

Toy Dataset. Topic Modeling // 100K (R=1) and 100K (R=5) following the design from the previous week
-Unable to perform SVM on a 160 K sample  (80K (R=1) and 80K (R=5)) on the local host
-Able to replicated the behaviour seen in the previous week using a 20K sample (10K (R=1) and 10K (R=5))




````
SMV // embeddings+ mean distances (euclidean, minkowski, cosine); (20Kx306)
#                  precision    recall  f1-score   support

# 0 = rating of 1   0.898404  0.905098  0.901739      2550
# 1 = rating of 5   0.900452  0.893469  0.896947      2450

#        accuracy                       0.899400      5000
#       macro avg   0.899428  0.899284  0.899343      5000
#    weighted avg   0.899408  0.899400  0.899391      5000

````
vs.

````
SMV // embeddings only (20Kx300)

#                  precision    recall  f1-score   support

# 0 = rating of 1   0.899065  0.904706  0.901876      2550
# 1 = rating of 5   0.900164  0.894286  0.897215      2450

#        accuracy                       0.899600      5000
#       macro avg   0.899615  0.899496  0.899546      5000
#    weighted avg   0.899604  0.899600  0.899593      5000


````
vs.

````
SMV // euclidean distance only (20Kx2)


#                  precision    recall  f1-score   support

# 0 = rating of 1   0.808314  0.823529  0.815851      2550
# 1 = rating of 5   0.812656  0.796735  0.804617      2450

#        accuracy                       0.810400      5000
#       macro avg   0.810485  0.810132  0.810234      5000
#    weighted avg   0.810442  0.810400  0.810346      5000


````

The embeddings play the major contrinuting role. 

An MLPClassifier model using the embeddings+ mean distances (euclidean, minkowski, cosine) performs slightly worse than SVM

`````


#                  precision    recall  f1-score   support

# 0 = rating of 1   0.888714  0.886275  0.887493      2550
# 1 = rating of 5   0.881970  0.884490  0.883228      2450

#        accuracy                       0.885400      5000
#       macro avg   0.885342  0.885382  0.885360      5000
#    weighted avg   0.885409  0.885400  0.885403      5000




`````

# Cross-Category testing

Using the previously obtained topics, we obtain a 20K sample (10K (R=1) and 10K (R=5)) from a new data set--Baby products. 
We find two things:
1) The embeddings are  still the major contributing feature
2) Embedding + three types of distances produce  precision, recall, f1-score in the 80s. 

Finding #2 shows worse performance on test data from the new category. It is worth nothing the statistics is still quite impressive though.


````

SMV // embeddings+ mean distances (euclidean, minkowski); (20Kx304)
#                  precision    recall  f1-score   support

# 0 = rating of 1   0.883057  0.897255  0.890099      2550
# 1 = rating of 5   0.891241  0.876327  0.883721      2450

#        accuracy                       0.887000      5000
#       macro avg   0.887149  0.886791  0.886910      5000
#    weighted avg   0.887067  0.887000  0.886974      5000
````

vs. 
````
SMV // mean minkowski distances  (20Kx2)

#                  precision    recall  f1-score   support

# 0 = rating of 1   0.777262  0.788235  0.782710      2550
# 1 = rating of 5   0.776305  0.764898  0.770559      2450

#        accuracy                       0.776800      5000
#       macro avg   0.776784  0.776567  0.776635      5000
#    weighted avg   0.776793  0.776800  0.776756      5000
````
vs
````
#SMV // embeddings only

# 0 = rating of 1   0.882285  0.896471  0.889321      2550
# 1 = rating of 5   0.890411  0.875510  0.882898      2450

#        accuracy                       0.886200      5000
#       macro avg   0.886348  0.885990  0.886109      5000
#    weighted avg   0.886267  0.886200  0.886174      5000



```


