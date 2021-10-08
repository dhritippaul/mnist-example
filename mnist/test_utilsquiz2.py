import sklearn
from skimage import transform
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split

def preprocessing(image,rescale_factor):
    resized_images = []
    for img in image:
        resized_images.append(transform.rescale(img,rescale_factor,anti_aliasing=False))
    return resized_images

def create_split(data,target,train_size,valid_size,test_size):
    assert  train_size == 0.70
    assert test_size == 0.20
    assert val_size == 0.10
    assert (train_size + val_size + test_size) ==1
    train_X,test_X,train_Y,test_Y = train_test_split(data,target,test_size=test_size + valid_size,shuffle=False)
    val_X,test_X,val_Y,test_Y = train_test_split(test_X,test_Y,test_size =((10/3)*test_size/(test_size+val_size)),shuffle = False)

    return train_X,train_Y,test_X,test_Y,val_X,val_Y

def get_acc(model,X,Y):
    pred = model.predict(X)
    acc = metrics.accuracy_score(Y,pred)
    return acc

def test_csplit():
    digits = datasets.load_digits()
    images = digits.images
    random_test = random.sample(range(0,1500),25)
    data = []
    Y = []
    for i in random_test:
        data.append(digits.images[i])
        Y.append(digits.target[i])
    data= np.array(data)
    Y = np.array(Y)
    train_size == 0.70
    test_size == 0.20
    val_size == 0.10
    train_X,test_X,train_Y,test_Y = train_test_split(data,target,test_size=test_size + valid_size,shuffle=False)
    val_X,test_X,val_Y,test_Y = train_test_split(test_X,test_Y,test_size =((10/3)*test_size/(test_size+val_size)),shuffle = False)
    print(len(train_X))
    print(len(test_X))
    print(len(val_X))
    length = len(data)
    train_split = len(train_X)
    test_split = len(test_X)
    val_split =len(val_X)
    assert  train_size == int(0.70*length)
    assert test_size == (0.20*length)
    assert val_size == (0.10*length)
    assert(train_split + val_split + test_split) == length
    
test_csplit()
