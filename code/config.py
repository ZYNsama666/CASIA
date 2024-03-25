dict_path = "dictionary.txt"
# train_path = "../source/Dataset/People_Daily/example.train"
train_path = "../source/Dataset/People_Daily/LOC.train"
validation_path = "../source/Dataset/People_Daily/example.dev"
test_path = "../source/Dataset/People_Daily/example.test"

OneHot_len = 500  # one-hot向量长度
other = OneHot_len - 3  # 其他词在one-hot向量中下标
front = OneHot_len - 2  # 句首在one-hot向量中下标
tail = OneHot_len - 1  # 句尾在one-hot向量中下标

lr = 1e-3  # 学习率
batch_size = 500  # batch
epoch = 10000  # 训练次数
val_time = 500  # 每训练多少次验证一次
weight = 0.15   # 计算loss
