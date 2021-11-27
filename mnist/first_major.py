import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import random
digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

hyperparameter_values =[0.00001,0.001,0.05]
c_value=[0.001,1,10]


for run in range(1,4):
    
    
    print("***Hyperparameters**** \t ******Run",run,"******")
        
    print("Gamma   C(value)    Train(acc,f1_score) \t Test(acc,f1_score)  \t Val(acc,f1_score) ")
    for j in range(0,3):
        hyper_random = random.choice(hyperparameter_values)
        cval = random.choice(c_value)


        clf_train = svm.SVC(gamma = hyper_random, C =cval)
        clf_test = svm.SVC(gamma =hyper_random,C =cval)
        clf_val = svm.SVC(gamma = hyper_random,C =cval)

        def create_split(data,target,train_size,valid_size,test_size):
            X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=test_size + valid_size,shuffle=False)
            X_val,X_test,y_val,y_test = train_test_split(X_test,y_test,test_size =(10/3)*test_size,shuffle = False)
            return X_train,y_train,X_test,y_test,X_val,y_val
        X_train,y_train,X_test,y_test,X_val,y_val = create_split(data = data ,target = digits.target,train_size=0.7,valid_size=0.15,test_size=0.15)

        clf_train.fit(X_train, y_train)
        clf_test.fit(X_test, y_test)
        clf_val.fit(X_val, y_val)

        predicted_train = clf_train.predict(X_train)
        predicted_test = clf_train.predict(X_test)
        predicted_val = clf_train.predict(X_val)
        
        acc_train = metrics.accuracy_score(predicted_train,y_train)
        acc_test = metrics.accuracy_score(predicted_test,y_test)
        acc_val = metrics.accuracy_score(predicted_val,y_val)


        f1_score_train = metrics.f1_score(y_train, predicted_train, average='macro')
        f1_score_test = metrics.f1_score(y_test, predicted_test, average='macro')
        f1_score_val = metrics.f1_score(y_val, predicted_val, average='macro')

        
        
        print(hyper_random,"  ",cval,"\t\t",format(acc_train, ".3f"),format(acc_test, ".3f"),"\t\t",format(acc_val, ".3f") , format(f1_score_train, ".3f"),"\t\t\t",format(f1_score_test, ".3f"),format(f1_score_val, ".3f"))

print("Yes the accuracy and f1_scores are changing.Written briefly in README.me") 