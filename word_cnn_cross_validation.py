from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from preprocess_word_cnn import convert_urls_to_vector
from evaluating_indicator import metric_F1score,metric_precision,metric_recall
# 设定随机数种子
seed = 7
np.random.seed(seed)
#词典大小
#word_size=0
#url分词后的长度
url_len = 300
#嵌入层输出字符维度
out_dimension = 64
word_size=185

#k折交叉验证，使用StratifiedKFold类将数据集分成10个子集
kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
#所有的epochs(1,2,3...20)训练准确率
taccuracy_count=[]
#所有的epochs(1,2,3...20)测试准确率，精准率，F1值，召回率
vaccuracy_count = []
F1_count=[]
precision_count=[]
recall_count=[]



def create_model():
    model = Sequential()
    # 构建嵌入层
    model.add(Embedding(word_size, out_dimension, input_length=url_len))
    # 1维度卷积层
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
    x,y,unique_word = convert_urls_to_vector(file_names, is_phishing)
    #word_size=unique_word
    # 创建模型 for scikit-learn
    #model = KerasClassifier(build_fn=create_model, epochs=1, batch_size=10)
    # 10折交叉验证
   # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # 训练并验证模型，每个epochs（包含150个epoch）后都有验证集去验证模型，总共进行k=10次。
    #results = cross_val_score(model, x, y, cv=kfold)
    #print(results.mean())

    for epoch in range(1, 31):

        vaccuracy = []

        taccuracy = []

        precision=[]

        recall=[]

        F1=[]

        for train, validation in kfold.split(x, y):
            model=create_model()
            history = model.fit(x[train], y[train], epochs=epoch, batch_size=100).history

            tac = history["accuracy"][epoch - 1]

            vac = model.evaluate(x[validation], y[validation], verbose=0)

            taccuracy.append(tac)

            vaccuracy.append(vac[1])
            precision.append(vac[2])
            recall.append(vac[3])
            F1.append(vac[4])

        taccuracy_count.append(np.mean(taccuracy))
        vaccuracy_count.append(np.mean(vaccuracy))
        precision_count.append(np.mean(precision))
        recall_count.append(np.mean(recall))
        F1_count.append(np.mean(F1))
    f = open(r"E:\daima-sx\test2\result\evaluating_indicator_word", "w+")
    f.writelines('taccuracy_count'+str(taccuracy_count)+'\n')
    f.writelines('vaccuracy_count'+str(vaccuracy_count)+'\n')
    f.writelines('precision_count' + str(precision_count)+'\n')
    f.writelines('recall_count' + str(recall_count)+'\n')
    f.writelines('F1_count' + str(F1_count) + '\n')
    f.close()


if __name__ == '__main__':
    main()
