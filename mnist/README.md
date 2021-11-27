

## ----------Refer compare.py----------
# SVM classifier is used here.
![alt text](https://github.com/dhritippaul/mnist-example/blob/feature/major/mnist/major1.png)
Accuracy and f1_score is printed for each train , test and validation for values of hyperparameter.
We got the best value in Run 2 and Run 3 in validation set. With increase in run iterations , random values of both the parameters are selected.
Yes, we do see variations in the results with each change in hyperparameter values and iterations.
One interesting thing to be noticed is the best value(performance of the classifier) we get is when value of Gamma is small (0.001 in our case) and value of C is large (10 in our case).
Good hyperparameter : Small Gamma , Large C.
Bad hyperparameter : Small C, Large Gamma.


