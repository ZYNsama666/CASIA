if target_Y == 0:
    if score0 >= score1 and score0 >= score2:  # 实际是0，预测为0
        T0 = T0 + 1
    if score1 >= score0 and score1 >= score2:  # 实际是0，预测为1
        F10 = F10 + 1
    if score2 >= score0 and score2 >= score1:  # 实际是0，预测为2
        F20 = F20 + 1
elif target_Y == 1:
    if score0 >= score1 and score0 >= score2:  # 实际是1，预测为0
        F01 = F01 + 1
    if score1 >= score0 and score1 >= score2:  # 实际是1，预测为1
        T1 = T1 + 1
    if score2 >= score0 and score2 >= score1:  # 实际是1，预测为2
        F21 = F21 + 1
else:
    if score0 >= score1 and score0 >= score2:  # 实际是2，预测为0
        F02 = F02 + 1
    if score1 >= score0 and score1 >= score2:  # 实际是2，预测为1
        F12 = F12 + 1
    if score2 >= score0 and score2 >= score1:  # 实际是2，预测为2
        T2 = T2 + 1

pre0 = T0 / T0 + F10 + F20  # 所有实际是0的猜对了多少
pre1 = T1 / T1 + F01 + F21
pre2 = T2 / T2 + F02 + F12
precision = (pre0 + pre1 + pre2) / 3

rec0 = T0 / T0 + F01 + F02  # 所有预测是0的猜对了多少
rec1 = T1 / T1 + F10 + F12
rec2 = T2 / T2 + F20 + F21
recall = (rec0 + rec1 + rec2) / 3

f1_0 = 2 * pre0 * rec0 / (pre0 + rec0)
f1_1 = 2 * pre1 * rec1 / (pre1 + rec1)
f1_2 = 2 * pre2 * rec2 / (pre2 + rec2)
f1 = (f1_0 + f1_1 + f1_2) / 3

print("loss={}".format(loss))  # 输出loss
print("acc={}\n".format(accuracy))
print("pre@0={.3f} pre@1={.3f} pre@2={.3f} precision={.3f}".format(pre0, pre1, pre2, precision))
print("rec@0={.3f} rec@1={.3f} rec@2={.3f} recall={.3f}".format(rec0, rec1, rec2, recall))
print("f1@0={.3f} f1@1={.3f} f1@2={.3f} f1-measure={.3f}".format(f1_0, f1_1, f1_2, f1))