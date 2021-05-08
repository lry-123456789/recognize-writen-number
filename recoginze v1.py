#使用keras定义简单的神经网络
from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671)            #重复性设置

#网络和训练
NB_EPOCH=2
BATCH_SIZE=128
VERBOSE=1
NB_CLASSES=10   #输出个数等于数字个数
OPTIMIZER=SGD() #SGD优化器
N_HIDDEN=128
VALIDATION_SPLIT=0.2 #训练集中用作验证集的比例

#数据：混合并划分训练集和测试集的数据
#
(X_train,y_train),(X_test,y_test)=mnist.load_data()
#X_train是60000行28*28的数据，变形为60000*784
RESHAPED=784
#
X_train=X_train.reshape(60000,RESHAPED)
X_test=X_test.reshape(10000,RESHAPED)
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
#归一化
#
X_train/=255
X_test/=255
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
#将类向量转化为二值类别矩阵
Y_train=np_utils.to_categorical(y_train, NB_CLASSES)
Y_test=np_utils.to_categorical(y_test, NB_CLASSES)
#10个输出
#最后是softmax激活函数
model=Sequential()
model.add(Dense(NB_CLASSES,input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.summary()
#下面为测试部分
#编译模型
model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,
              metrics=['accuracy'])
#训练模型
history=model.fit(X_train,Y_train,
                  batch_size=BATCH_SIZE,epochs=NB_EPOCH,
                  verbose=VERBOSE,validation_split=VALIDATION_SPLIT)
#测试模型
score=model.evaluate(X_test,Y_test,verbose=VERBOSE)
print("Test score:",score[0])
print("Test accuracy:",score[1])
