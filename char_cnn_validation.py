from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from preprocess_char_cnn import convert_urls_to_vector
from evaluating_indicator import metric_F1score,metric_precision,metric_recall
import matplotlib.pyplot as plt
import time
# 设定随机数种子
seed = 7
np.random.seed(seed)
#唯一字符数量
unique_char = 95
#url长度
max_char = 300
#嵌入层输出字符维度
out_dimension = 64
taccuracy_count=[]
vaccuracy_count = []
F1_count=[]
precision_count=[]
recall_count=[]

def create_model():
    model = Sequential()
    model.add(Embedding(unique_char, out_dimension, input_length=max_char))
    model.add(Conv1D(filters=200, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',metric_precision,metric_recall,metric_F1score])
    model.summary()
    return model

def main():
    file_names = ["dataset/phishing_url.txt", "dataset/cc_1_first_9617_urls"]
    is_phishing = [True, False]
    x,y = convert_urls_to_vector(file_names, is_phishing)
    model=create_model()
    start = time.clock()
    history = model.fit(x, y, epochs=3, batch_size=100,validation_split=0.2).history
    print("time:",time.clock() - start)
    plt.plot(history['loss'])
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    # plt.plot(history['metric_precision'])
    plt.ylabel('accuracy/loss')
    plt.xlabel('epochs')
    # plt.legend(['train_loss', 'train-accuracy','trian_precision'])
    plt.show()
if __name__ == '__main__':
    main()
