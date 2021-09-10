
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


n_samples = len(digits.images)
print("Shape of original image : ",digits.images[0].shape)
print("Image taken here is of 16*16 with 70%training data and 30%testing data")
print("Gamma_Value          Accuracy")



for user_gamma in np.arange(0.002, 0.020, 0.002):
    new_image = np.zeros((n_samples,16,16))
    for i in range(0,n_samples):
        new_image[i] = resize(digits.images[i],(16,16),anti_aliasing=True)
    data = new_image.reshape((n_samples, -1))
    clf = svm.SVC(gamma=user_gamma)
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=False)
    clf.fit(X_train, y_train)  
    predicted = clf.predict(X_test)
    print(user_gamma ,"            ", metrics.accuracy_score(predicted , y_test))



