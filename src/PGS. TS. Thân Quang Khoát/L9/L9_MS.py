# Model Selection
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
from tqdm import tqdm

from sklearn.datasets import load_files
from pyvi import ViTokenizer

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, pair_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# load data
INPUT = 'data/news_vnexpress'
os.makedirs('images', exist_ok=True) # thư mực lưu các hình ảnh kết quả trong quá trình huấn luyện và đánh giá

# statistics
print('Các nhãn và số văn bản tương ứng trong dữ liệu')
print('---------------------------------------------')
n = 0
for label in os.listdir(INPUT):
    print(f'{label}: {len(os.listdir(os.path.join(INPUT, label)))}')
    n += len(os.listdir(os.path.join(INPUT, label)))

print('---------------------------------------------')
print(f'Tổng số văn bản: {n}')

data_train = load_files(container_path=INPUT, encoding='utf-8')

print('mapping:')
for i in range(len(data_train.target_names)):
    print(f'{data_train.target_names[i]} - {i}')

print('---------------------------------------------')
print(data_train.filenames[0:1])
print(data_train.target[0:1])
print(data_train.data[0:1])

print("\nTổng số văn bản: {}".format(len(data_train.filenames)))

# Tiền xử lý dữ liệu
# Chuyển dữ liệu văn bản thành dạng số
# load dữ liệu các stopwords
with open('data/vietnamese-stopwords.txt', encoding='utf-8') as f:
    stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords]
print(f'Số lượng stopwords: {len(stopwords)}')
print(stopwords[:10])

module_count_vector = CountVectorizer(stop_words=stopwords)
module_rf_preprocess = Pipeline([('vect', module_count_vector), ('tf', TfidfTransformer()),])

data_preprocessed = module_rf_preprocess.fit_transform(data_train.data, data_train.target)

print(f'\nSố lượng từ trong từ điển: {len(module_count_vector.vocabulary_)}')
print(f'Kích thước dữ liệu sau khi xử lý: {data_preprocessed.shape}')
print(f'Kích thước nhãn tương ứng: {data_train.target.shape}')

# Chia dữ liệu thành tập train và test (dùng hold-out)
p = 0.2
pivot = int(data_preprocessed.shape[0] * (1-0.2))
X_train, X_test = data_preprocessed[0:pivot], data_preprocessed[pivot:]
y_train, y_test = data_train.target[0:pivot], data_train.target[pivot:]

# lựa chọn (tối ưu) tham số
def cross_validation(estimator):
    _, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=10, n_jobs=-1, train_sizes=[1.0, ], scoring='accuracy')
    test_scores = test_scores[0]
    mean, std = test_scores.mean(), test_scores.std()
    return mean, std

def plot(title, xlabel, X, y, error, ylabel='Accuracy'):
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid()
    plt.ylabel(ylabel)

    plt.errorbar(X, y, error, linestyle='None', marker='o')

# Đánh giá hiệu quả của các kernel trong SVM
title = "thay đổi kernel, C = 1"
xlabel = "kernel"
X = []
y = []
error = []

for kernel in tqdm(['linear', 'poly', 'rbf', 'sigmoid']):
    # với mỗi kernel được chọn
    # thực hiện xây dựng mô hình, huấn luyện và đánh giá theo cross-validation
    text_clf = svm.SVC(kernel=kernel, C=1.0)
    mean, std = cross_validation(text_clf)
    X.append(kernel)
    y.append(mean)
    error.append(std)

# lưu kết quả ra file ảnh
plot(title, xlabel, X, y, error)
# plt.savefig('images/svm_change_kernel.png', bbox_inches='tight')
# plt.show()

# Đánh giá ảnh hưởng của tham số C trong SVM
title = "thay đổi C, kernel = linear"
xlabel = "C"
X = []
y = []
error = []

for C in tqdm([.1, 1.0, 2.0, 5.0, 10.0]):
    # với mỗi C được chọn
    # thực hiện xây dựng mô hình, huấn luyện và đánh giá theo cross-validation
    text_clf = svm.SVC(kernel='linear', C=C)
    mean, std = cross_validation(text_clf)
    X.append(C)
    y.append(mean)
    error.append(std)

plot(title, xlabel, X, y, error)
# plt.savefig('images/svm_change_C.png', bbox_inches='tight')
# plt.show()

# Đánh giá ảnh hưởng của độ đo trong Random Forest
title = "thay đổi criterion, n_estimators = 50"
xlabel = "criterion"
X = []
y = []
error = []

for criterion in tqdm(['gini', 'entropy']):
    # với mỗi criterion được chọn
    # thực hiện xây dựng mô hình, huấn luyện và đánh giá theo cross-validation
    text_clf = RandomForestClassifier(criterion=criterion, n_estimators=50)
    mean, std = cross_validation(text_clf)
    X.append(criterion)
    y.append(mean)
    error.append(std)

plot(title, xlabel, X, y, error)
# plt.savefig('images/RF_change_criterion.png', bbox_inches='tight')
# plt.show()

# Đánh giá ảnh hưởng của số cây trong Random Forest
title = "thay đổi n_estimators, criterion = gini"
xlabel = "n_estimators"
X = []
y = []
error = []

for n_estimators in tqdm([10, 50, 100, 300]):
    # với mỗi n_estimators được chọn
    # thực hiện xây dựng mô hình, huấn luyện và đánh giá theo cross-validation
    text_clf = RandomForestClassifier(criterion='gini', n_estimators=n_estimators)
    mean, std = cross_validation(text_clf)
    X.append(n_estimators)
    y.append(mean)
    error.append(std)

plot(title, xlabel, X, y, error)
# plt.savefig('images/RF_change_N.png', bbox_inches='tight')
# plt.show()

# so sánh các mô hình
svm_ = svm.SVC(kernel='linear', C=1.0)
rf = RandomForestClassifier(criterion='gini', n_estimators=100)

svm_.fit(X_train, y_train)
rf.fit(X_train, y_train)

print(f'SVM: {accuracy_score(y_test, svm_.predict(X_test))}')
print(f'RF: {accuracy_score(y_test, rf.predict(X_test))}')

ConfusionMatrixDisplay(svm_, X_test, y_test)
ConfusionMatrixDisplay(rf, X_test, y_test)