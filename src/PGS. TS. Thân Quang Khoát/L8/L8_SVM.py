# Phân loại sử dụng SVM
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.datasets import load_files
from pyvi import ViTokenizer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

# load data
DATA_PATH = "data/news_vnexpress/"
header = "%-20s%-30s" % ("Số lượng văn bản", "Nhãn")
print(header)
print("-"*30)
total = 0
for label in os.listdir(DATA_PATH):
    n = len(os.listdir(os.path.join(DATA_PATH, label)))
    total += n
    entry = "%-20d%-30s" % (n, label)
    print(entry)

print("-"*30)
print(f"Tổng số văn bản: {total}")

data_train = load_files(container_path=DATA_PATH, encoding="utf-8")
print(dir(data_train))

header = "%-6s %-10s" % ("ID", "Nhãn")
print(header)
print("-"*30)
for id, label in enumerate(data_train.target_names):
    print("%-6d %-10s" % (id, label))

print(data_train.data[0:2], end="\n\n")
print(data_train.filenames[0:2], end="\n\n")
print(data_train.target[0:2], end="\n\n")

print(data_train.data.__len__())
print(data_train.target.__len__())
print(data_train.filenames.__len__())

# tiền xử lý dữ liệu: đưa dữ liệu từ dạng text về dạng ma trận = TF-IDF
# load dữ liệu từ các stopwords
with open("data/vietnamese-stopwords.txt", encoding="utf-8") as f:
    stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords]
print(f"Tổng số lượng từ dừng: {len(stopwords)}")
print("Danh sách 10 từ dừng đầu tiên (từ không mang ý nghĩa phân loại):", stopwords[:10])
print()

module_count_vector = CountVectorizer(stop_words=stopwords)
module_rf_preprocess = Pipeline([('vect', module_count_vector), ('tfidf', TfidfTransformer())])
data_preprocessed = module_rf_preprocess.fit_transform(data_train.data, data_train.target)
print("5 từ đầu tiên trong từ điển:\n")
i = 0
for k, v in module_count_vector.vocabulary_.items():
    i += 1
    print(i, ":", (k, v))
    if i > 5:
        break

print()

# số chiều của dữ liệu
print(f"Số chiều của dữ liệu: {data_preprocessed.shape}")
print(f"Số từ trong từ điển: {len(module_count_vector.vocabulary_)}")

# chia dữ liệu thành 2 phần train_data và test_data
from sklearn.model_selection import ShuffleSplit

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, data_train.target, test_size=test_size)
print("Dữ liệu training =", X_train.shape, y_train.shape)
print("Dữ liệu testing =", X_test.shape, y_test.shape)

print("ID   Nhãn")
print("-" * 15)
for i in range(5):
    label_id = y_train[i]
    label_name = data_train.target_names[label_id]
    print(f"{i}    {label_name}")

# huấn luyện mô hình SVM trên tập train_data
print("- Training ...")

print("- Train size = {}".format(X_train.shape))
model = svm.SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

print("- model - train complete")

# đánh giá mô hình SVM trên tập test_data
from sklearn.metrics import accuracy_score
print("- Testing ...")
y_pred = model.predict(X_test)
print("- Acc = {}".format(accuracy_score(y_test, y_pred)))

# sử dụng model đã được huấn luyện để phán đoán 1 văn bản mới
news = ["Công_phượng ghi bàn cho đội_tuyển Việt_Nam"]
preprocessed_news = module_rf_preprocess.transform(news)
print(preprocessed_news, end="\n\n")
# phán đoán nhãn
pred = model.predict(preprocessed_news)
print(pred, data_train.target_names[pred[0]])

# Bài tập bổ sung
# Đánh giá các tham số của mô hình SVM: kernel, C
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
best_kernel = None
best_acc = 0.0

for kernel in kernels:
    model = svm.SVC(kernel=kernel, C=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Kernel: {kernel}, Acc: {acc}")
    if acc > best_acc:
        best_acc = acc
        best_kernel = kernel

print(f"Best kernel: {best_kernel}")

C_values = [0.1, 1.0, 5.0, 10.0]
best_C = None
best_acc = 0.0

for C in C_values:
    model = svm.SVC(kernel='linear', C=C)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"C: {C}, Acc: {acc}")
    if acc > best_acc:
        best_acc = acc
        best_C = C

print(f"Best C: {best_C}")

# phân loại số viết tay
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
target = digits.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size)

print("Dữ liệu training =", X_train.shape, y_train.shape)
print("Dữ liệu testing =", X_test.shape, y_test.shape)

