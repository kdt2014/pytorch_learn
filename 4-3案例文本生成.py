import torch
import re

# 构建词典
def build_vocab():

    fname = 'data/jaychou_lyrics.txt'

    # 1.文本数据清洗
    clean_sentences = []
    for line in open(fname, 'r', encoding='utf-8'):

        # 去除指定内容
        # 保留中文，数字和部分标点符号
        line = re.sub(r'[^\u4e00-\u9fa5 a-zA-Z0-9!?,]', '', line)
        # 连续空格替换成1个
        line = re.sub(r'[ ]{2,}', '', line)
        # 去除两侧空格、换行
        line = line.strip()
        # 去除单字的行
        if len(line) <= 1:
            continue

        # 去除重复行
        if line not in clean_sentences:
            clean_sentences.append(line)

    print(clean_sentences)




if __name__ == '__main__':
    build_vocab()
