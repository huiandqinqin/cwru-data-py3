# cwru-data-py3
## Describe:
Automatically download the cwru data set, and then divide it into training data set and test data set.  
Data is not enhanced.  
```python
python3.6, cwrudataset
```
自动下载cwru数据集，然后分训练数据集和测试数据集。  
数据并为作增强处理  

## How to use it?
```python 
from cwru_data_py3 import CWRU
from sklearn.ensemble import RandomForestClassifier
data = CWRU.CWRU("12DriveEndFault", "1797", 1024, -1)
X_train, y_train, X_test, y_test = data.X_train, data.y_train, data.X_test, data.y_test
##   rf_model  随机森林模型
rf_model = RandomForestClassifier(n_estimators= 300, max_features = "sqrt", n_jobs = -1, random_state = 38)
rf_model.fit(X_train, y_train)
```
## Arguments
CWRU has four arguments:
- exp: experiment, supporting "12DriveEndFault", "12FanEndFault", "48DriveEndFault"
- rpm: rpm during testing
- length: length of the signal slice, namely X_train.shape[1]
- directory： -1 means parent_dir， 1 means current_dir
  ``` python
  -1 means
  ---you_project_name
     ---A.py
  ---DataSet/XXX
   1 means 
   ---you_project_name
      ---A.py
      ---DataSet/
   ```
