# coding=UTF-8
from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import TensorBoard
import numpy as np
from preprocessing import augumentSimple
from sklearn.model_selection import train_test_split
import time
from use_augument_data.hyper_tunnning import *
from use_augument_data.my_utils.get_train_test import *






def wdcnn():
    def wdcnn1(filters, kernerl_size, strides, conv_padding, pool_padding, pool_size, BatchNormal):
        """wdcnn层神经元

        :param filters: 卷积核的数目，整数
        :param kernerl_size: 卷积核的尺寸，整数
        :param strides: 步长，整数
        :param conv_padding: 'same','valid'
        :param pool_padding: 'same','valid'
        :param pool_size: 池化层核尺寸，整数
        :param BatchNormal: 是否Batchnormal，布尔值
        :return: model
        """
        model.add(Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides,
                         padding=conv_padding, kernel_regularizer=l2(1e-4)))
        if BatchNormal:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=pool_size, padding=pool_padding))
        return model
    # 实例化序贯模型
    model = Sequential()
    # 搭建输入层，第一层卷积。因为要指定input_shape，所以单独放出来
    model.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same', kernel_regularizer=l2(1e-4),
                     input_shape=(1024,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    # 第二层卷积

    model = wdcnn1(filters=32, kernerl_size=3, strides=1, conv_padding='same',
                  pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
    # 第三层卷积
    model = wdcnn1(filters=64, kernerl_size=3, strides=1, conv_padding='same',
                  pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
    # 第四层卷积
    model = wdcnn1(filters=64, kernerl_size=3, strides=1, conv_padding='same',
                  pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
    # 第五层卷积
    model = wdcnn1(filters=64, kernerl_size=3, strides=1, conv_padding='valid',
                  pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
    # 从卷积到全连接需要展平
    model.add(Flatten())

    # 添加全连接层
    model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))
    # 增加输出层
    model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))

    # 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



def train_wdcnn(path):
    start = time.time()
    x_train, x_test, y_train, y_test = get_train_test_1dim(path)
    input_shape = x_train.shape[1:]
    model = wdcnn()
    # 开始模型训练
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, shuffle=True)
    # validation_data=(x_valid, y_valid),
    # 评估模型

    loss, accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
    print("测试集上的损失：", loss)
    print("模型上的正确率:", accuracy)
    end = time.time()
    print("本次消耗的时间为:" + str(end - start))
    return accuracy, end - start


if __name__ == '__main__':
    accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    i = 0
    ## 运行在四个转速下下
    if i > 10:
        temp_list = []
        for n in range(len(path_list)):
            for i in range(10):
                temp = train_wdcnn(path_list[n])
                accuracy_list[path_list[n][-3:]].append(temp[0])
                time_list[path_list[n][-3:]].append(temp[1])
    ## 运行在某一个转速下
    else:
        for i in range(4):
            temp = train_wdcnn(path_list[0])
            accuracy_list[path_list[0][-3:]].append(temp[0])
            time_list[path_list[0][-3:]].append(temp[1])
    print(accuracy_list)
    print(time_list)