import codecs
import random
# 先针对LOC
train_path = "../source/Dataset/People_Daily/example.train"
after_pre_train_path = "../source/Dataset/People_Daily/LOC.train"
training_set = codecs.open(train_path, 'r', 'utf-8')
after_pre_training_set = codecs.open(after_pre_train_path, 'w', 'utf-8')
temp_list = []
flag = 0


def write(list):
    for item in list:
        s = item[0] + " " + item[1] + "\n"
        after_pre_training_set.write(s)
    after_pre_training_set.write('\n')


for word_tag in training_set.readlines():
    word = word_tag.strip().split(" ")

    if word[0] == "":
        if flag == 0:  # 如果整个句子就没有B或者I，那么就有一定概率不要了。
            test = random.uniform(0, 1)
            if test > 0.9:
                write(temp_list)

        if flag == 1:  # 只要这个句子中包含
            write(temp_list)
        temp_list.clear()
        flag = 0
    else:
        temp_list.append(word)
        if word[1] == "B-LOC" or word[1] == "I-LOC":
            flag = 1
