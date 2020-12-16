from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Dropout
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from preprocess_char_word_cnn_lstm import convert_urls_to_vector
from keras.preprocessing import sequence
from keras.datasets import imdb
from evaluating_indicator import metric_F1score,metric_precision,metric_recall
# 设定随机数种子
seed = 7
np.random.seed(seed)

#url分词后的长度
url_len = 300
#嵌入层输出字符维度
out_dimension = 64
word_size=121+1


def create_model():
    model = Sequential()
    # 构建嵌入层
    model.add(Embedding(word_size, out_dimension, input_length=url_len))
    # 1维度卷积层
    model.add(Conv1D(filters=200, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=100))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def main():
    file_names = ["dataset/phishing_url.txt", "dataset/cc_1_first_9617_urls"]
    is_phishing = [True, False]
    x,y = convert_urls_to_vector(file_names, is_phishing)
    # 创建模型 for scikit-learn
    #model = KerasClassifier(build_fn=create_model, epochs=1, batch_size=100)
    # 10折交叉验证
   # kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    # 训练并验证模型，每个epochs（包含150个epoch）后都有验证集去验证模型，总共进行k=10次。
   # results = cross_val_score(model, x, y, cv=kfold)
   # print(results.mean())
    #model = create_model()
    # #训练模型
    #model.fit(x, y, batch_size=128, epochs=2)
    #scores = model.evaluate(x, y, verbose=2)
    #print("scores:",scores)
    #print('Accuracy: %.2f%%' % (scores[1] * 100))

    #(x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=5000)

    # 限定数据集的长度
    #x_train = sequence.pad_sequences(x_train, maxlen=url_len)
    #x_validation = sequence.pad_sequences(x_validation, maxlen=url_len)

    # 生产模型并训练模型
    model = create_model()
    model.fit(x, y, batch_size=128, epochs=20)



if __name__ == '__main__':
    main()
