# -*- coding: utf-8 -*-
# @Date    :2024-12-25 21:06:34
# @Author  :SongWei (songweiof@gmail.com)
# @Link    :https://github.com/Finder12345
# @Version :python 3.11
# @Software:Sublime Text




from collections import defaultdict
import math

# 构建有向无环图 (DAG)
def build_dag(sentence, dictionary):
    n = len(sentence)
    dag = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n + 1):
            word = sentence[i:j]
            if word in dictionary:
                # 使用对数概率
                log_prob = math.log(dictionary[word])
                dag[i].append((j, log_prob))  # 存储 (终点, 对数概率)
    print(dag)
    return dag

# 动态规划计算最优路径
def find_best_cut_log(sentence, dictionary):
    n = len(sentence)
    dag = build_dag(sentence, dictionary)
    
    # DP数组：dp[i] 表示从第i个字符到末尾的最大对数概率
    dp = [-float('inf')] * (n + 1)
    dp[n] = 0  # 终点的对数概率为0（乘法单位）
    
    # 路径记录：用于回溯最优切分
    path_record = [-1] * (n + 1)
    
    # 从后向前进行动态规划
    for i in range(n - 1, -1, -1):
        for j, log_prob in dag[i]:
            if dp[i] < dp[j] + log_prob:
                dp[i] = dp[j] + log_prob
                path_record[i] = j
    
    # 回溯最优切分路径
    best_path = []
    start = 0
    while start != -1 and start < n:
        end = path_record[start]
        best_path.append(sentence[start:end])
        start = end
    
    return best_path, math.exp(dp[0])  # 返回最优切分和总概率（从对数还原）

# 测试代码
dictionary = {
    "经常": 0.1, "经": 0.05, "有": 0.1, "常": 0.001,
    "有意见": 0.1, "歧": 0.001, "意见": 0.2, "分歧": 0.2,
    "见": 0.05, "意": 0.05, "见分歧": 0.05, "分": 0.1
}
sentence = "经常有意见分歧"

# 动态规划找最优切分
best_cut, max_prob = find_best_cut_log(sentence, dictionary)

# 输出结果
print("最优切分方案:", " | ".join(best_cut))
print("总概率:", max_prob)

