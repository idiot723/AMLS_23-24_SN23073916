import medmnist
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score


def import_A():
  data = np.load('./dataset/pneumoniamnist.npz')

  x_train_3d = data['train_images']
  y_train = data['train_labels']
  x_test_3d = data['test_images']
  y_test = data['test_labels']

  x_train = x_train_3d.reshape((x_train_3d.shape[0], -1)) #reduce dimention from 3d to 2d
  x_test = x_test_3d.reshape((x_test_3d.shape[0], -1)) #reduce dimention from 3d to 2d

  return x_train,y_train,x_test,y_test

def KNN(x_train,y_train,x_test,y_test):
    k_range = range(1,10)
    tot_score = []

    for k in k_range:
      #train model
      knn = KNeighborsClassifier(n_neighbors=k)
      knn.fit(x_train, y_train)
      #test model
      Y_pred = knn.predict(x_test)
      score=metrics.accuracy_score(y_test,Y_pred)
      tot_score.append(score)

    k_best = k_range[tot_score.index(max(tot_score))]
    print("The best k is:",k_best,"the accuracy is", max(tot_score))



   
