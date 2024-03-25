from __future__ import unicode_literals
import torch.nn.functional as f
import codecs
import math
import torch
import random


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
    set_len = 0
    word_list.append(-1)
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
        set_len = set_len + 1
    return set_len  # 返回这一部分样本数量 作为后续生成的得分矩阵的一个维度


def evaluation(word_list, tag_list, length):  # 用于计算loss 查全率 查准率 准确率和f1-measure
    # _type中1是训练集 2是验证集 3是测试集
    global best_theta
    global best_f1
    loss = 0  # 损失函数 反映是否过拟合 若基本处于下降状态 则说明没有过拟合
    TP = TN = FP = FN = 0  # 后面每一项都有解释
    for word_num in range(1, length - 2):
        now = word_list[word_num]
        if now == -1:  # 跳过句首
            continue
        X = get_x(word_list, word_num)
        Y = tag_list[word_num]  # 当前词的标签（0 1 2->O B I）

        tmp = f.softmax(torch.mm(theta.T, X), dim=0)
        score = float(tmp[Y])
        loss += -math.log(score)

        score0 = float(tmp[0])
        score1 = float(tmp[1])
        score2 = float(tmp[2])
        # 认为BI为正例，O为负例
        if Y == 0:
            if score0 >= score1 and score0 >= score2:  # 实际是0，预测为0
                TN = TN + 1
            else:
                FN = FN + 1
        elif Y == 1:
            if score1 >= score0 and score1 >= score2:  # 实际是1，预测为1
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if score2 >= score0 and score2 >= score1:  # 实际是2，预测为2
                TP = TP + 1
            else:
                FP = FP + 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    if f1 > best_f1:  # 维护每次验证集的theta中得到了loss最好的loss
        best_theta = theta.clone()
        best_f1 = f1

    print("loss={}".format(loss))  # 输出loss
    print("TP={} FN={} FP={} TN={}".format(TP, FN, FP, TN))
    print("precision={} recall={} acc={}\nf1-measure={}\n".format(precision, recall, accuracy, f1))
    # 输出查准率 查全率 准确率 f1-measure
    return f1


def get_x(word_list, word_num_):  # 将三个词向量连接起来生成测试集
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


def train_and_test():
    global theta
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
    print("-------------训练+验证-------------")  # 在测试集上进行测试

    for t in range(0, run_time+1):  # 在验证集上进行测试
        if t % 200 == 0:  # 每训练一定次进行一次验证
            print("{}/{}\n".format(t, run_time), end='')
            evaluation(validation_word, validation_tag, validation_len)
        sum_ = torch.zeros(3 * OneHot_len, 3)

        for i in range(0, batch_size):
            sample = random.randint(1, training_len-2)  # SGD
            if training_word[sample] == -1:  # 用于区分句首句尾的不训练
                continue
            X = get_x(training_word, sample)  # 1500*1
            Y = training_tag[sample]
            target = torch.zeros(3, 1)
            target[Y] = 1
            output = f.softmax(torch.mm(theta.T, X), dim=0)  # 3*1500*1500*1=3*1
            loss = -(output.log() * target).sum()  # 交叉熵自动求导
            theta.retain_grad()
            loss.backward()
            sum_ = theta.grad + sum_
            theta.grad.zero_()
        theta = theta - lr * sum_/batch_size  # 梯度下降

    print("\n-------------test-------------\n")  # 在测试集上进行测试
    theta = best_theta.clone()
    evaluation(test_word, test_tag, test_len)

    training_set.close()
    validation_set.close()
    test_set.close()


if __name__ == "__main__":
    dict_path = "dictionary.txt"
    train_path = "../../source/Dataset/People_Daily/example.train"
    validation_path = "../../source/Dataset/People_Daily/example.dev"
    test_path = "../../source/Dataset/People_Daily/example.test"
    OneHot_len = 500  # one-hot向量长度
    other = OneHot_len - 3  # 其他词在one-hot向量中下标
    front = OneHot_len - 2  # 句首在one-hot向量中下标
    tail = OneHot_len - 1  # 句尾在one-hot向量中下标

    lr = 1e-1  # 学习率
    batch_size = 200  # SGD
    run_time = 5000  # 训练次数

    dictionary = {}  # 高频词字典 构造词->one-hot下标的映射

    theta = torch.randn(3 * OneHot_len, 3)  # theta 1500 * 3维 随机初始值
    best_theta = torch.randn(3 * OneHot_len, 3)  # best_theta 1500 * 3维 用于储存最好的theta值
    best_f1 = -1  # 维护出现最好的f1值，用于判断更新best_theta
    theta.requires_grad = True

    train_and_test()
