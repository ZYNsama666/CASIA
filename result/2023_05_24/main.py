from __future__ import unicode_literals
import torch.nn.functional as f
from module import *
import codecs
import torch
import random


def get_model():
    my_model = MLP(OneHot_len=OneHot_len)
    return my_model, optim.SGD(my_model.parameters(), lr=lr)


def build_dict(filename):  # 读入高频词，建立高频词词典
    dictionary_file = codecs.open(filename, 'r', 'utf-8')
    counter = 0
    for sentence in dictionary_file.readlines():
        words = sentence.strip().split(" ")
        for word in words:
            dictionary[word] = counter
            counter = counter + 1


def build_list(set_, word_list, tag_list):
    # 对三个样本进行处理 将词处理成高频词词典中编号 标签处理成0（其他词） 1（B） 2（I）
    word_list.append(-1)  # 数据集最开始先塞一个“头”
    tag_list.append(-1)
    for word_tag in set_.readlines():
        word = word_tag.strip().split(" ")
        if word[0] == "":
            word_list.append(-1)
            tag_list.append(-1)
        else:
            if dictionary.__contains__(word[0]):  # 标注为对应的向量所在维数
                word_list.append(dictionary[word[0]])
            else:
                word_list.append(other)  # 标记为其他

            if word[1] == "B-LOC":  # 标签标注序号
                tag_list.append(1)
            elif word[1] == "I-LOC":
                tag_list.append(2)
            else:
                tag_list.append(0)
    return len(word_list)  # 返回这一部分样本数量 作为后续生成的得分矩阵的一个维度


def get_x(word_list, word_num_):  # 将三个词向量连接起来生成测试向量
    now_ = word_list[word_num_]
    X = torch.zeros(3 * OneHot_len, 1)  # 初始化一个1500维的向量
    # 特判头没有前一向量 尾没有后一向量 剩下的直接记录
    if word_list[word_num_ - 1] == -1:  # 前面是-1表示是头
        pre_ = front
    else:
        pre_ = word_list[word_num_ - 1]

    if word_list[word_num_ + 1] == -1:  # 后面是-1表示是尾
        nxt_ = tail
    else:
        nxt_ = word_list[word_num_ + 1]

    X[pre_] = 1
    X[now_ + OneHot_len] = 1
    X[nxt_ + 2 * OneHot_len] = 1
    return X


def evaluation(word_list, tag_list, length):  # 用于计算loss 查全率 查准率 准确率和f1-measure
    loss = 0
    TP = TN = FP = FN = 0

    for word_num in range(1, length - 2):
        now = word_list[word_num]
        if now == -1:  # 跳过句首句尾分隔信息
            continue
        X = get_x(word_list, word_num)
        target_Y = tag_list[word_num]
        Y = torch.zeros(1, 3)
        Y[0][target_Y] = 1
        predict = f.softmax(model(X.T), dim=1)

        # loss += (predict[0][target_Y]-1) * (predict[0][target_Y]-1)
        loss = loss + f.cross_entropy(predict, Y)

        score0 = float(predict[0][0])
        score1 = float(predict[0][1])
        score2 = float(predict[0][2])

        # 认为BI为正例，O为负例
        if target_Y == 0:
            if score0 >= score1 and score0 >= score2:  # 实际是负例0，预测正确
                TN = TN + 1
            else:  # 实际是负例0，预测错误
                FP = FP + 1
        elif target_Y == 1:
            if score1 >= score0 and score1 >= score2:  # 实际是正例1，预测为1
                TP = TP + 1
            else:  # 实际是正例1，预测错误
                FN = FN + 1
        else:
            if score2 >= score0 and score2 >= score1:  # 实际是正例2，预测为2
                TP = TP + 1
            else:  # 实际是正例2，预测错误
                FN = FN + 1
    if TP + FP == 0:
        print("TP={} FN={} FP={} TN={}\n".format(TP, FN, FP, TN))
        return
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    print("loss={}".format(loss))  # 输出loss
    print("TP={} FN={} FP={} TN={}".format(TP, FN, FP, TN))
    print("precision={} recall={} acc={}\nf1-measure={}\n".format(precision, recall, accuracy, f1))


def train_and_test():
    print("-------------train-------------")  # 在测试集上进行测试

    for t in range(epoch + 1):  # 训练集训练，在验证集上进行测试
        if t % val_time == 0:  # 每训练一定次进行一次验证
            print("{}/{}\n".format(t, epoch), end='')
            # evaluation(training_word, training_tag, training_len)  # 训练集
            evaluation(validation_word, validation_tag, validation_len)  # 验证集

        batch_start = random.randint(1, training_len - batch_size - 1)
        loss = 0
        for i in range(0, batch_size):
            index = batch_start + i
            if training_word[index] == -1:  # 用于区分句首句尾的不训练
                continue
            X = get_x(training_word, index)  # 1500*1
            target_Y = training_tag[index]
            Y = torch.zeros(1, 3)
            Y[0][target_Y] = 1

            predict = f.softmax(model(X.T), dim=1)
            # loss += (predict[0][target_Y]-1) * (predict[0][target_Y]-1)
            temp_loss = f.cross_entropy(predict, Y)
            if target_Y == 0:
                rand_num = random.uniform(0, 1)
                if rand_num < weight:   # 丢弃一些是O的结果
                    loss = loss + temp_loss
            else:
                loss = loss + temp_loss

        loss.backward()
        opt.step()
        opt.zero_grad()

    print("\n-------------test-------------\n")  # 在测试集上进行测试
    evaluation(test_word, test_tag, test_len)

    training_set.close()
    validation_set.close()
    test_set.close()


if __name__ == "__main__":
    dict_path = "dictionary.txt"
    # train_path = "../source/Dataset/People_Daily/example.train"
    train_path = "../source/Dataset/People_Daily/after_pre.train"
    validation_path = "../source/Dataset/People_Daily/example.dev"
    test_path = "../source/Dataset/People_Daily/example.test"

    OneHot_len = 500  # one-hot向量长度
    other = OneHot_len - 3  # 其他词在one-hot向量中下标
    front = OneHot_len - 2  # 句首在one-hot向量中下标
    tail = OneHot_len - 1  # 句尾在one-hot向量中下标

    lr = 1e-3  # 学习率
    batch_size = 200  # batch
    epoch = 1000  # 训练次数
    val_time = 100  # 每训练多少次验证一次
    weight = 0.05

    model, opt = get_model()
    dictionary = {}  # 高频词字典 构造词->one-hot下标的映射

    # 三部分样本的词 以在字典中的值的方式记录
    training_word = []
    validation_word = []
    test_word = []

    # 三部分样本的标签 以0 1 2的方式记录O B I
    training_tag = []
    validation_tag = []
    test_tag = []

    build_dict(dict_path)
    training_set = codecs.open(train_path, 'r', 'utf-8')
    validation_set = codecs.open(validation_path, 'r', 'utf-8')
    test_set = codecs.open(test_path, 'r', 'utf-8')

    training_len = build_list(training_set, training_word, training_tag)  # 初始化 处理数据集
    validation_len = build_list(validation_set, validation_word, validation_tag)
    test_len = build_list(test_set, test_word, test_tag)

    print("基本信息统计：")
    print("训练集大小:{} 验证集大小：{} 测试集大小：{}".format(training_len, validation_len, test_len))

    train_and_test()
