from preprocessing import augumentSimple
from sklearn.model_selection import train_test_split
from deepforest import CascadeForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import time
from use_augument_data.hyper_tunnning import *
from use_augument_data.my_utils.get_train_test import get_row_data



def train_deep_forest(path):
    start = time.time()
    # load data
    X, Y = get_row_data(path)
    ## chang shape
    X = X.reshape(X.shape[0], 1024)
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    model = CascadeForestClassifier(n_jobs=-1, n_estimators=4, n_trees=300, max_layers=20)
    model.fit(x_train, y_train)

    # train and evaluate
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred) * 100  # classification accuracy
    print("accuracy", acc)

    end = time.time()
    print("本次消耗的时间为:" + str(end - start))

    # show confusion matrix
    test_confu_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(test_confu_matrix, annot=True,
                xticklabels=fault_types, yticklabels=fault_types, cmap="Blues", cbar=False)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.show()
    return acc, end - start

if __name__ == '__main__':
    accuracy_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    time_list = {'0HP': [], '1HP': [], '2HP': [], '3HP': []}
    # 获得data 下的数据
    temp_list = []
    for n in range(len(path_list)):
        for i in range(1):
            temp = train_deep_forest(path_list[n])
            accuracy_list[path_list[n][-3:]].append(temp[0])
            time_list[path_list[n][-3:]].append(temp[1])
    print(accuracy_list)
    print(time_list)
