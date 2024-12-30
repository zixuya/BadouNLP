import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    try:
        with open(path, encoding="utf8") as f:
            for line in f:
                sentence = line.strip()
                sentences.add(" ".join(jieba.cut(sentence)))
        print("获取句子数量：", len(sentences))
    except FileNotFoundError:
        print(f"文件 {path} 未找到，请检查路径。")
        return set()
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"C:\Users\Junmming\A01_File\AI_data\A05周\week5 词向量及文本向量\model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 计算类内距离
    for label, sentences in sentence_label_dict.items():
        vectors_in_cluster = sentences_to_vectors(sentences, model)
        distance_sum = 0
        count = 0
        for i in range(len(vectors_in_cluster)):
            for j in range(i + 1, len(vectors_in_cluster)):
                diff = vectors_in_cluster[i] - vectors_in_cluster[j]
                distance = np.sqrt(np.sum(diff ** 2))
                distance_sum += distance
                count += 1
        mean_distance = distance_sum / count if count > 0 else 0
        sentence_label_dict[label] = (mean_distance, sentences)

    sorted_clusters = sorted(sentence_label_dict.items(), key=lambda x: x[1][0])

    for rank, (label, (distance, sentences)) in enumerate(sorted_clusters):
        print(f"Rank {rank + 1}, cluster {label}, mean distance: {distance:.4f}")
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")
