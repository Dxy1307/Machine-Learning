# dữ liệu gồm n văn bản phân vào 10 chủ đề khác nhau. Cần biểu diễn mỗi văn bản dưới dạng một vector số thể hiện cho nội dung của văn bản đó
import os
import matplotlib.pyplot as plt
import numpy as np

from pyvi import  ViTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files

INPUT = 'data/news_vnexpress'
os.makedirs("images", exist_ok=True) # thư mục lưu các hình ảnh kết quả trong quá trình huấn luyện và đánh giá

# statistics
print('Các nhãn và số văn bản tương ứng trong dữ liệu')
print('----------------------------------------------')
n = 0
for label in os.listdir(INPUT):
    print(f'{label}: {len(os.listdir(os.path.join(INPUT, label)))}')
    n += len(os.listdir(os.path.join(INPUT, label)))

print('-------------------------')
print(f'Tổng số văn bản: {n}')

# load data
data_train = load_files(container_path=INPUT, encoding='utf-8')
print('mapping:')
for i in range(len(data_train.target_names)):
    print(f'{data_train.target_names[i]} - {i}')

print('---------------------------')
print(data_train.filenames[0:1])
# print(data_train.data[0:1])
print(data_train.target[0:1])
print(data_train.data[0:1])

print("\nTổng số văn bản: {}" .format(len(data_train.filenames)))

# chuyển dữ liệu dạng text về ma trận (n x m) bằng TF-IDF
# load dữ liệu các stopwords
with open("data/vietnamese-stopwords.txt", encoding="utf-8") as f:
    stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords]
print(f"Số lượng stopwords: {len(stopwords)}")
print(stopwords[:10])

# chuyển hóa dữ liệu text về dạng vector TF
#   - loại bỏ từ dừng
#   - sinh từ điển
module_count_vector = CountVectorizer(stop_words=stopwords)
model_rf_preprocess = Pipeline([('vect', module_count_vector), ('tfidf', TfidfTransformer()), ])
# hàm tực hiện chuyển đổi dữ liệu text -> dữ liệu số dạng ma trận
# Input: Dữ liệu 2 chiều dạng numpy.array, mảng nhãn id dạng numpy.array
data_preprocessed = model_rf_preprocess.fit_transform(data_train.data, data_train.target)

print(f"\nSố lượng từ trong từ điển: {len(module_count_vector.vocabulary_)}")
print(f"Kích thước dữ liệu sau khi xử lý: {data_preprocessed.shape}")
print(f"Kích thước nhãn tương ứng: {data_train.target.shape}")

X = data_preprocessed
Y = data_train.target
# print(X.shape)
# print(Y.shape)
print(X[100].toarray())
print(Y[100])
sum(sum(X[100].toarray() != 0))
print(X[100])
