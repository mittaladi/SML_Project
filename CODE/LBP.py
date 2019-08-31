import os
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes as nb
from sklearn.svm import SVC

label = []
arr = []
lab = 0

folders = os.listdir("SML_dataset\Dataset_CK+TFEID")
for folder in folders:
    file = os.listdir("SML_dataset\Dataset_CK+TFEID\\"+folder)
    for i in file:
        arr.append("SML_dataset\Dataset_CK+TFEID\\" + folder + "\\" + i)
        label.append(lab)
    lab += 1
    
data = []
for url in arr:
    data.append(cv2.resize(cv2.imread(url , 0), (int(200), int(200))))
    
plt.style.use("ggplot")
(fig, ax) = plt.subplots()
fig.suptitle("Local Binary Patterns")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel bucket")
#
#cv2.imwrite("IMG_20181208_2220211.jpg", features.astype("uint8"))
#
b = ax.hist(features.ravel(), normed=True, bins=256, range=(0, 256))
#ax.set_xlim([0, 256])
#ax.set_ylim([0, 0.030])
## save figure
#fig.savefig('IMG_20181208_222021.jpg')   # save the figure to file
#plt.show()


flatten_dataset = []


for image in data:
    features = feature.local_binary_pattern(image , 10, 5, method="default")
    
    flatten_features = features.flatten()
    flatten_dataset.append(flatten_features)
    
#    b = ax.hist(features.ravel(), normed=True, bins=256, range=(0, 256))
#    ax.set_xlim([0, 256])
#    ax.set_ylim([0, 0.030])
#    
flatten_dataset = np.array(flatten_dataset)
x_train, x_test, y_train, y_test = train_test_split(
        flatten_dataset, label, test_size=0.3, random_state=42)


logis_clf = LogisticRegression().fit(x_train , y_train)
logis_acc = logis_clf.score(x_test,y_test)
print('accuracy for logistic regression' , logis_accuracy)

naive_clf = nb.GaussianNB().fit(x_train,y_train)
naive_acc = naive_clf.score(x_test,y_test)
print('accuracy for naive bayes' , naive_accuracy)


svm_clf = SVC(decision_function_shape='ovo').fit(x_train,y_train)
svm_acc = svm_clf.score(x_test,y_test)
print('accuracy for support vector machine' , svm_accuracy)
    
    
pca = PCA(.95)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)

logis_clf = LogisticRegression().fit(pca_train , y_train)
naive_clf = nb.GaussianNB().fit(pca_train,y_train)
svm_clf = SVC(decision_function_shape='ovo').fit(pca_train,y_train)

logis_accuracy = logis_clf.score(pca_test,y_test)
print('accuracy for logistic regression' , logis_accuracy)

naive_accuracy = naive_clf.score(pca_test,y_test)
print('accuracy for naive bayes' , naive_accuracy)

svm_accuracy = svm_clf.score(pca_test,y_test)
print('accuracy for support vector machine' , svm_accuracy)

classifiers = 3
without_pca = (logis_acc,naive_acc,svm_acc)
with_pca = (logis_accuracy,naive_accuracy,svm_accuracy)
    
fig, ax = plt.subplots()
index = np.arange(classifiers)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, without_pca, bar_width, alpha=opacity,
                             color='b', label='Without PCA')
 
rects2 = plt.bar(index + bar_width, with_pca, bar_width,
                     alpha=opacity, color='g', label='With PCA')
 
plt.xlabel('Classifiers')
plt.ylabel('Accuracies')
plt.title('Scores by person')
plt.xticks(index + bar_width, ('Logistic', 'Naive Bayes', 'SVM'))
plt.legend()
 
plt.tight_layout()
plt.show()
    
    