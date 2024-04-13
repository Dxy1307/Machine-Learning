# Dự đoán khả năng tiến triển của bệnh tiểu đường thông qua các chỉ số sinh lý của cơ thể.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math
import seaborn as sns

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# đọc dữ liệu huấn luyện
with open("Course/data/diabetes_train.pkl", 'rb') as f:
    # data = f.read()
    # diabetes_train = pickle.loads(data)
    diabetes_train = pickle.load(f)
print("Số chiều input:", diabetes_train['data'].shape)
print("Số chiều target y tương ứng:", diabetes_train['target'].shape)
print()

print('2 mẫu dữ liệu đầu tiên')
print('input:', diabetes_train['data'][:2])
print('input:', diabetes_train['target'][:2])

# xây dựng mô hình hồi quy sử dụng sklearn
regr = linear_model.LinearRegression()

# huấn luyện mô hình
regr.fit(diabetes_train['data'], diabetes_train['target'])
print("[w1, ... w_n =", regr.coef_)
print("w0 =", regr.intercept_)

# dự đoán các mẫu dữ liệu trong tập test
# phán đoán
# đọc dữ liệu test
# dữ liệu test có cấu trúc giống dữ liệu huấn luyện nhưng số lượng mẫu chỉ là 42
with open('Course/data/diabetes_test.pkl', 'rb') as f:
    diabetes_test = pickle.load(f)

# thực hiện phán đoán cho dữ liệu mới
diabetes_y_pred = regr.predict(diabetes_test['data'])

# kiểm tra chất lượng phán đoán
df = pd.DataFrame(data=np.array([diabetes_test['target'], diabetes_y_pred, abs(diabetes_test['target'] - diabetes_y_pred)]).T, columns=["y thực tế", 'y dự đoán', 'Lệch'])

#in ra 5 phán đoán đầu tiên
print(df.head(5))

# sử dụng độ đo RMSE(căn bậc 2 trung bình bình phương lỗi)
rmse = math.sqrt(mean_squared_error(diabetes_test['target'], diabetes_y_pred))
print(f'RMSE = {rmse}')

# phân phối các dự đoán đầu ra của mô hình
sns.displot(diabetes_y_pred, )
plt.show()