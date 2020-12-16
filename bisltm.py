from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Dropout,Bidirectional
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from preprocess_char_word_cnn_lstm import convert_urls_to_vector
from evaluating_indicator import metric_F1score,metric_precision,metric_recall
import matplotlib.pyplot as plt
import time
# 设定随机数种子
seed = 7
np.random.seed(seed)
url_len = 300
out_dimension = 64
word_size=121+1
taccuracy_count=[]
vaccuracy_count = []
F1_count=[]
precision_count=[]
recall_count=[]
v_loss_count=[]
t_loss_count=[]
def create_model():
    model = Sequential()
    model.add(Embedding(word_size, out_dimension, input_length=url_len))
    model.add(Conv1D(filters=200, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    lstm=LSTM(128)
    model.add(lstm)
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',metric_precision,metric_recall,metric_F1score])
    model.summary()
    return model


def main():
    file_names = ["dataset/phishing_url.txt", "dataset/cc_1_first_9617_urls"]
    is_phishing = [True, False]
    x,y = convert_urls_to_vector(file_names, is_phishing)
    model =create_model()
    start = time.clock()
    history=model.fit(x, y, batch_size=100, epochs=30,validation_split=0.2).history
    print("time:", time.clock() - start)
    # 创建模型 for scikit-learn
    model = KerasClassifier(build_fn=create_model, epochs=1, batch_size=100)
    # 10折交叉验证
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    # 训练并验证模型，每个epochs（包含150个epoch）后都有验证集去验证模型，总共进行k=10次。
    results = cross_val_score(model, x, y, cv=kfold)
    print(results.mean())
    # model = create_model()
    #训练模型
    model.fit(x, y, batch_size=128, epochs=2, verbose=2)
    scores = model.evaluate(x, y, verbose=2)
    print("scores:",scores)
    print('Accuracy: %.2f%%' % (scores[1] * 100))
    for epoch in range(1, ):
        # 测试准确率
        vaccuracy = []
        # 训练准确率
        taccuracy = []
        #测试精准率
        precision=[]
        #测试召回率
        recall=[]
        #测试F1值
        F1=[]
        # train和validation为数据分割后的（数组）索引
        for train, validation in kfold.split(x, y):
            model=create_model()
            print(x[train], y[train])
            history = model.fit(x[train], y[train], epochs=epoch, batch_size=100).history
            # 获取训练数据epoch的准确率
            tac = history["accuracy"][epoch - 1]
            # 评估模型，通过设置verbose（啰嗦的、冗长的）为0，关闭evaluate()函数的详细输出
            vac = model.evaluate(x[validation], y[validation], verbose=0)
            # 输出评估结果
            print('%s: %.2f%%' % (model.metrics_names[1], vac[1]))
            # 获取k折中每折的训练正确率
            taccuracy.append(tac)
            # 获取k折中每折的测试正确率
            vaccuracy.append(vac[1])
            precision.append(vac[2])
            recall.append(vac[3])
            F1.append(vac[4])
        # 输出训练准确率均值和测试准确率均值
        print(np.mean(taccuracy), np.mean(vaccuracy))
        # 获取每个epoch的训练准确率均值和测试准确率均值
        taccuracy_count.append(np.mean(taccuracy))
        vaccuracy_count.append(np.mean(vaccuracy))
        precision_count.append(np.mean(precision))
        recall_count.append(np.mean(recall))
        F1_count.append(np.mean(F1))
    f = open(r"E:\daima-sx\test2\result\evaluating_indicator_my", "w+")
    f.writelines('taccuracy_count'+str(taccuracy_count)+'\n')
    f.writelines('vaccuracy_count'+str(vaccuracy_count)+'\n')
    f.writelines('precision_count' + str(precision_count)+'\n')
    f.writelines('recall_count' + str(recall_count)+'\n')
    f.writelines('F1_count' + str(F1_count) + '\n')
    f.close()

if __name__ == '__main__':
    main()
