from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten, AveragePooling2D
from keras.models import Sequential
import time
from use_augument_data.hyper_tunnning import *
from use_augument_data.my_utils.get_train_test import get_train_test_1dim

def zhao_1d():
    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=5, activation='relu',padding='same', input_shape=(1024, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D())

    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='same' ))
    model.add(BatchNormalization())
    model.add(MaxPooling1D())

    model.add(Flatten())

    # model.add(Dense(units=1024, activation='relu'))
    #
    # model.add(Dropout(0.5))

    model.add(Dense(units=256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(units=10, activation='softmax'))

    # 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_zhao_1d(path):
    start = time.time()
    x_train, x_test, y_train, y_test = get_train_test_1dim(path)
    input_shape = x_train.shape[1:]
    model = zhao_1d()
    history = model.fit(x_train, y_train, epochs=20)
    loss, accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)
    model.summary()
    print("测试集上的损失：", loss)
    print("模型上的正确率:", accuracy)
    end = time.time()
    print("本次消耗的时间为:" + str(end - start))
    return accuracy, end - start

if __name__ == '__main__':
    accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    # 获得data 下的数据
    i = 0
    if i > 10:
        temp_list = []
        for n in range(len(path_list)):
            for i in range(10):
                temp = train_zhao_1d(path_list[n])
                accuracy_list[path_list[n][-3:]].append(temp[0])
                time_list[path_list[n][-3:]].append(temp[1])
        print(accuracy_list)
        print(time_list)
    else:
        for i in range(10):
            temp = train_zhao_1d(path_list[1])
            accuracy_list[path_list[0][-3:]].append(temp[0])
            time_list[path_list[0][-3:]].append(temp[1])
    print(accuracy_list)
    print(time_list)