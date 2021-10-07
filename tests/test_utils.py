from utils import Experiment_loaded
from sklearn import datasets
import random
import numpy as np


digits = datasets.load_digits()

def test_small_daat_overfitting():
    images = digits.images
    randomlist = random.sample(range(0,1796),10)
    data = []
    Y = []
    for i in randomlist:
        data.append(digits.images[i])
        Y.append(digits.target[i])
    data = np.array(data)
    Y = np.array(Y)
    data = data/255
    data = data.reshape((10,-1))
    train_metrics = Experiment_loaded(train_X=data,train_Y=Y,val_X=data,val_Y=Y)
    assert max(train_metrics) > 0.90