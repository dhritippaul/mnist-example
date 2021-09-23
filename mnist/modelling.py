from math import gamma
import sklearn
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,svm,metrics
from sklearn.model_selection import train_test_split
from joblib import dump,load
import os

digits = datasets.load_digits()

n_sample = len(digits.images)
data = digits.images.reshape((n_sample,-1))
model_candidate = []
resize_images_size = [0.25,0.5,1,2,4,8]
hyperparameter_values =[0.00001,0.0001,0.001,0.05,1,2,5,10]
print("\t\tResize \t\tDataset \t\tgamma_value \t\tAccuracy")
os.mkdir('models')
for i in range(len(resize_images_size)):
    resized_images = []
    
    for img in digits.images:
        resized_images.append(transform.rescale(img,resize_images_size[i],anti_aliasing=False))
    for j in range(len(hyperparameter_values)):
        resized_images = np.array(resized_images)
        data = resized_images.reshape((n_sample,-1))
        model = svm.SVC(gamma = hyperparameter_values[j])
        test_size = 0.15
        val_size = 0.15
        train_X,test_X,train_Y,test_Y = train_test_split(data,digits.target,test_size=0.3,shuffle=False)
        val_X,test_X,val_Y,test_Y = train_test_split(test_X,test_Y,test_size =0.5,shuffle = False)
        model.fit(train_X,train_Y)
        predict_val = model.predict(val_X)
        acc_val = metrics.accuracy_score(y_pred= predict_val,y_true = val_Y)
        if acc_val<0.20:
            print("Skipping for {}".format(hyperparameter_values[j]))
            continue
        candidate = {
            "acc_valid" : acc_val,
            "gamma" : hyperparameter_values[j],
        }
        model_candidate.append(candidate)
        output_folder = "./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(test_size,val_size,resize_images_size[i],hyperparameter_values[j])
        os.mkdir(output_folder)
        dump(model,os.path.join(output_folder,'model.joblib'))
        print("Saving Model for {}".format(hyperparameter_values[j]))
    max_valid_acc_model_candidate = max(model_candidate,key=lambda x:x["acc_valid"])
    best_model_folder ="./models/tt_{}_val_{}_rescale_{}_gamma_{}".format(test_size,val_size,resize_images_size[i],max_valid_acc_model_candidate["gamma"])
    path = os.path.join(best_model_folder,'model.joblib')
    model = load(path)
    predict_test = model.predict(test_X)
    acc_test = metrics.accuracy_score(y_pred= predict_test,y_true = test_Y)
    print("\t\t ",resize_images_size[i] ,"\t\tValidation Set","\t\t",hyperparameter_values[j],"\t\t",acc_val*100)
    print("\t\t ",resize_images_size[i] ,"\t\tTest Set","\t\t",hyperparameter_values[j],"\t\t",acc_test*100)
