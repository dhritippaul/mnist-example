import random
import numpy as np
from utils import prev_exp
from sklearn import datasets
digits = datasets.load_digits()

def exp():
    images = digits.images
    req_list = random.sample(range(0,1500),25)
    values = []
    L = []
    values = np.array(values)
    L = np.array(L)
    for i in req_list:
        values.append(digits.images[i])
        L.append(digits.target[i])
    values = values/255
    values = values.reshape((10,-1))
    exp_data = prev_exp(train_X=values,train_Y=L,val_X=values,val_Y=L)
    assert max(exp_data) > 0.90
