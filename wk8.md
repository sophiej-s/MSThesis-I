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



`````



