import numpy as np
import pandas as pd
# đọc dữ liệu vào pandas dataframe
my_data = pd.read_csv("data/drug200.csv", delimiter=",")
print(my_data[0:5])
# tiền xử lý dữ liệu
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
from sklearn import preprocessing
# mô hình Random Forest chỉ xử lý vs số thực nên cần chuyển đổi dữ liểu dạng categorial về số thực
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = le_Chol.transform(X[:, 3])

print(X[0:5])

# chuẩn hóa thuộc tính Na_to_K về mean = 0 và std = 1
mean = X[:, -1].mean()
std = X[:, -1].std()
X[:, -1] = (X[:, -1] - mean) / std
print(X[0:5])

# dữ liệu target về loại thuốc phản ứng
y = my_data["Drug"]
print(y[0:5])

# đưa y về dạng số thực
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])
y = le_sex.transform(y)

print(y[0:5])