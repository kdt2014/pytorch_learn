import torch
import torch.nn as nn
import jieba

if __name__ == '__main__':

    text = '北京冬奥的进度条已经过半，不少外国的运动员在完成自己的比赛后踏上归途。'

    # 1. 分词
    words = jieba.lcut(text)
    print(words)

    # 2.构建词表
    index_to_word = {} # 索引编号确定词
    word_to_index = {} # 词确定索引编号
    # 去重
    unique_word = list(set(words))

    for idx, word in enumerate(unique_word):
        index_to_word[idx] = word
        word_to_index[word] = idx

    # 3 构建词嵌入层
    embed = nn.Embedding(num_embeddings=len(words), embedding_dim=4)

    # 4 将文本转换为词向量表示
    # print(index_to_word[0])
    # print(embed(torch.tensor(0)))
    # 将句子数值化
    for word in words:
        # 获得word的索引
        idx = word_to_index[word]
        word_vec = embed(torch.tensor(0))
        print('%3s\t' % word, word_vec)
