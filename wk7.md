# 

# Design:
![slide 001](https://user-images.githubusercontent.com/20401990/157798424-caf9410b-4745-42a2-9fcd-e1b11ca6b2d3.jpeg)

# Results using SVM (with 'rbf' kernel)

Both embeddings and the distance:
````
#                  precision    recall  f1-score   support

# 0 = rating of 1   0.902610  0.907472  0.905034      4269
# 1 = rating of 5   0.906131  0.901205  0.903662      4231

#        accuracy                       0.904353      8500
#       macro avg   0.904370  0.904339  0.904348      8500
#    weighted avg   0.904362  0.904353  0.904351      8500


````



Embeddings only:
`````
#                  precision    recall  f1-score   support

# 0 = rating of 1   0.902541  0.906770  0.904651      4269
# 1 = rating of 5   0.905486  0.901205  0.903340      4231

#        accuracy                       0.904000      8500
#       macro avg   0.904014  0.903988  0.903996      8500
#    weighted avg   0.904007  0.904000  0.903998      8500


``````



Distance only:
``````
#                  precision    recall  f1-score   support

# 0 = rating of 1       0.81      0.83      0.82      4269
# 1 = rating of 5       0.82      0.81      0.81      4231

#        accuracy                           0.82      8500
#       macro avg       0.82      0.82      0.82      8500
#    weighted avg       0.82      0.82      0.82      8500



``````





