T0 = T1 = T2 = F10 = F20 = F01 = F21 = F02 = F12 = 0
if target_Y == 0:
    if max(score0, score1, score2) == score0:  # 实际是0，预测为0
        T0 = T0 + 1
    elif max(score0, score1, score2) == score1:  # 实际是0，预测为1
        F10 = F10 + 1
    else:  # 实际是0，预测为2
        F20 = F20 + 1
elif target_Y == 1:
    if max(score0, score1, score2) == score0:  # 实际是1，预测为0
        F01 = F01 + 1
    elif max(score0, score1, score2) == score1:  # 实际是1，预测为1
        T1 = T1 + 1
    else:  # 实际是1，预测为2
        F21 = F21 + 1
else:
    if max(score0, score1, score2) == score0:  # 实际是2，预测为0
        F02 = F02 + 1
    elif max(score0, score1, score2) == score1:  # 实际是2，预测为1
        F12 = F12 + 1
    else:  # 实际是2，预测为2
        T2 = T2 + 1

pre0 = T0 / (T0 + F01 + F02 + 1)  # 所有实际是0的猜对了多少
pre1 = T1 / (T1 + F10 + F12 + 1)
pre2 = T2 / (T2 + F20 + F21 + 1)
precision = (pre0 + pre1 + pre2) / 3

rec0 = T0 / (T0 + F10 + F20 + 1)  # 所有预测是0的猜对了多少
rec1 = T1 / (T1 + F01 + F21 + 1)
rec2 = T2 / (T2 + F02 + F12 + 1)
recall = (rec0 + rec1 + rec2) / 3

f1_0 = 2 * pre0 * rec0 / (pre0 + rec0 + 1)
f1_1 = 2 * pre1 * rec1 / (pre1 + rec1 + 1)
f1_2 = 2 * pre2 * rec2 / (pre2 + rec2 + 1)
f1 = (f1_0 + f1_1 + f1_2) / 3

accuracy = (T0 + T1 + T2) / (F10 + F20 + F01 + F21 + F02 + F12)

print("loss={} acc={}".format(loss, accuracy))  # 输出loss
print("pre@O={:.5f} pre@B={:.5f} pre@I={:.5f} precision={:.5f}".format(pre0, pre1, pre2, precision))
print("rec@O={:.5f} rec@B={:.5f} rec@I={:.5f} recall={:.5f}".format(rec0, rec1, rec2, recall))
print("f1@O={:.5f} f1@B={:.5f} f1@I={:.5f} f1-measure={:.5f}\n".format(f1_0, f1_1, f1_2, f1))




# 第二种
