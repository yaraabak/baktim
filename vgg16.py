import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.datasets import fashion_mnist

# the model
def pretrained_model(img_shape, num_classes):
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    keras_input = Input(shape=img_shape, name = 'image_input')
    

    output_vgg16_conv = model_vgg16_conv(keras_input)

    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(20, activation='relu', name='fc1')(x)
    x = Dense(20, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    

    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return pretrained_model


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


x_train = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in x_train]
x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')

x_test = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in x_test]
x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

model = pretrained_model(x_train.shape[1:], len(set(y_train)))
hist = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), verbose=1)
test_loss,test_acc=model.evaluate(x_test,y_test)
print('test loss',test_loss)
print('test accuracy',test_acc)