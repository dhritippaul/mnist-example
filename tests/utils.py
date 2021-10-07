import sklearn
from sklearn.model_selection import train_test_split
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump,load
import os
from numpy.core.getlimits import _fr1
from sklearn import datasets,svm,metrics


def preprocessing(image,rescale_factor):
    resized_images = []
    for img in image:
        resized_images.append(transform.rescale(img,rescale_factor,anti_aliasing=False))
    return resized_images

def create_split(data,target,train_size,valid_size,test_size):
    train_X,test_X,train_Y,test_Y = train_test_split(data,target,test_size=test_size + valid_size,shuffle=False)
    val_X,test_X,val_Y,test_Y = train_test_split(test_X,test_Y,test_size =(10/3)*test_size,shuffle = False)

    return train_X,train_Y,test_X,test_Y,val_X,val_Y

def get_acc(model,X,Y):
    pred = model.predict(X)
    acc = metrics.accuracy_score(Y,pred)
    f1 = metrics.f1_score(y_true=Y,y_pred=pred,average=None)
    return {"acc": acc, "f1": f1 }

def get_random_acc(y):
    return max(np.bincount(y))/len(y)

def prev_exp(train_X,train_Y,val_X,val_Y):
    acc_list = []
    gamma= [0.0001,0.001,1e-05]
    for i in range(len(gamma)):
        folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(0.15,0.15,1,gamma[i])
        path = os.path.join(folder,'model.joblib')
        model = load(path)
        acc_val = get_acc(model=model,X=train_X,Y=train_Y)
        acc_list.append(acc_val['acc'])
    return acc_list

def main_exp(train_X,train_Y,val_X,val_Y,gamma,output_model_path):
    rand_val_acc = get_random_acc(val_Y)

    model = svm.SVC(gamma = gamma)

    model.fit(train_X,train_Y)

    metrics_valid = get_acc(model= model,X = val_X,Y = val_Y)
    
    if metrics_valid["acc"] < rand_val_acc:
        print("Skipping for {}".format(gamma))
        return None
    
    output_folder = os.path.dirname(output_model_path)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    dump(model,output_model_path)
    return metrics_valid

