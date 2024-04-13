# Random forest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# load dữ liệu từ file
my_data = pd.read_csv("data/drug200.csv", delimiter=",")
my_data[0:5]
my_data.shape

# tiền xử lý dữ liệu
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])  # chuyển giới tính thành số
X[:, 1] = le_sex.transform(X[:, 1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])  # chuyển huyết áp thành số
X[:, 2] = le_BP.transform(X[:, 2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])  # chuyển cholesterol thành số
X[:, 3] = le_Chol.transform(X[:, 3])

X[0:5]
y = my_data["Drug"]
y[0:5]

# Cài đặt mô hình Random Forest
from sklearn.model_selection import train_test_split    # chia dữ liệu thành 2 phần: train và test  (80% và 20%)
# Hàm train_test_split sẽ chia dữ liệu thành 4 phần: X_trainset, X_testset, y_trainset, y_testset
# Hàm train_test_split cần các tham số đầu vào: X, y, test_size=0.3, random_state=3
# X và y là các ma trận và vector cần chia, test_size là tỉ lệ tập test và random_state để loại bỏ tính ngẫu nhiên
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print(X_trainset.shape)
print(y_trainset.shape)
print(X_testset.shape)
print(y_testset.shape)

# huấn luyện mô hình
# tạo một đối tượng RandomForestClassifier, cần chỉ rõ số cây tương ứng
drugTree = RandomForestClassifier(n_estimators=100)
# Huấn luyện mô hình trên tập X_trainset và y_trainset
drugTree.fit(X_trainset, y_trainset)

# dự đoán
predTree = drugTree.predict(X_testset)
# so sánh
print("Nhãn phán đoán")
print(predTree[0:5])
print("Nhãn đúng")
print(list(y_testset)[0:5])

# Đánh giá thử mô hình
# import metrics từ sklearn để kiểm tra độ chính xác của mô hình trên tập test
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# đánh giá ảnh hưởng của các thuộc tính với nhãn
print(list(my_data.columns)[0:-1])
print(list(drugTree.feature_importances_))