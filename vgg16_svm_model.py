
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
from keras.datasets import fashion_mnist 
import cv2

(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

train_X = [cv2.cvtColor(cv2.resize(i, (224,224)), cv2.COLOR_GRAY2BGR) for i in train_X]
train_X = np.concatenate([arr[np.newaxis] for arr in train_X]).astype('float32')

test_X = [cv2.cvtColor(cv2.resize(i, (224,224)), cv2.COLOR_GRAY2BGR) for i in test_X]
test_X = np.concatenate([arr[np.newaxis] for arr in test_X]).astype('float32')


train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

SIZE=224
N_ch=3
input_shape =(SIZE, SIZE, N_ch)
baseModel = VGG16(weights="imagenet", 
                  input_shape=(input_shape))

model = Model(inputs=baseModel.input, outputs=baseModel.get_layer('fc1').output)

X_train_feat_vgg = model.predict(train_X)
X_test_feat_vgg = model.predict(test_X)


knn = KNeighborsClassifier(n_neighbors =1)
knn.fit(X_train_feat_vgg,train_Y_one_hot)
pred=knn.predict(X_test_feat_vgg)

print("Accuracy: ",accuracy_score(test_Y_one_hot, pred))