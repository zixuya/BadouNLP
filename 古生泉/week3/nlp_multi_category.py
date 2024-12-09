# coding:utf8

import torch
import torch.nn as nn
import random
import numpy as np

"""
实现一个多分类的NLP任务
"""


# 定义模型
class Multi_Category(nn.Module):
    def __init__(self, dim, vocab, len_vocab):  # 每个字符的维度  字符集  每组文字的长度及几分类
        super(Multi_Category, self).__init__()  # 初始化
        self.enbedd = nn.Embedding(len(vocab), dim, padding_idx=0)  # enbedding层：字符集个数  dim：每个字符的维度
        self.pool = nn.AvgPool1d(len_vocab)  # 池化层  每组文字的长度
        self.linear = nn.Linear(len_vocab, len_vocab)  # enbedding层的维度  几分类
        self.rnn_liner = nn.RNN(dim, len_vocab, batch_first=True)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.enbedd(x)  # 传入embedding层
        y_pred = y_pred.transpose(1, 2)
        y_pred = self.pool(y_pred)  # 传入池化层
        y_pred = y_pred.squeeze()
        y_pred, _ = self.rnn_liner(y_pred)  # 传入线性层
        y_pred = self.linear(y_pred)  # 传入线性层
        # print(y_pred)
        if y is not None:
            loss = self.loss(y_pred, y)
            return loss
        else:
            return y_pred


# 构建词表
def build_vocab():
    char = ("深拷贝虽然能保证对象的独立性")
    char_set = {}
    count = 1
    char_set['pad'] = 0
    for i in char:
        if i in char_set:
            continue
        char_set[i] = count
        count += 1
    char_set['unk'] = len(char_set)
    return char_set


# 构建随机生成的语句
def build_sample(build_vocab, len_vocab):
    list_vocab = []
    zero = np.zeros(len_vocab)
    for i in range(len_vocab):
        list_vocab.append(random.choice(list(build_vocab.keys())))
    for k, j in enumerate(list_vocab):
        if '独' == j:
            zero[k] = 1
    list_vo = [build_vocab.get(i, build_vocab['unk']) for i in list_vocab]
    return list_vo, zero


# 构建数据集
def build_data(number, build_vocab, len_vocab):
    x = []
    y = []
    for i in range(number):
        x_i, y_i = build_sample(build_vocab, len_vocab)
        x.append(x_i)
        y.append(y_i)
    return torch.LongTensor(np.array(x)), torch.FloatTensor(np.array(y))


def evaluate(model, vocab):
    model.eval()
    test_sample_num = 1000
    len_vocab = 6
    x, y = build_data(test_sample_num, vocab, len_vocab)  # 生成训练数据
    print("本次预测集中共有%d个样本" % (test_sample_num))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(torch.argmax(y_p)) == float(torch.argmax(y_t)):
                correct += 1  # 判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 300  # 训练轮数
    batch_size = 100  # 每次训练数据样本个数
    train_sample = 10000  # 每轮训练的样本总数

    char_dim = 20  # 每个字的维度
    len_vocab = 6  # 样本文本长度
    lraening_rate = 0.001  # 学习率

    vocab = build_vocab()  # 构建字符集
    data_x, data_y = build_data(train_sample * batch_size, vocab, len_vocab)  # 生成训练数据
    modle = Multi_Category(char_dim, vocab, len_vocab)  # 定义模型
    optim = torch.optim.Adam(modle.parameters(), lr=lraening_rate)  # 寻找优化器

    for epoch in range(epoch_num):
        modle.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x = data_x[batch * batch_size: (batch + 1) * batch_size]
            y = data_y[batch * batch_size: (batch + 1) * batch_size]
            loss = modle(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(modle, vocab)
    torch.save(modle.state_dict(), "modelmulti.bin")
    # print(len(vocab))
    # print(build_dat)



if __name__ == '__main__':
    main()
