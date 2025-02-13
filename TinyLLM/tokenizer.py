"""



method1:
    python实现的bpe算法，封装为tokenizer
or

method2:
    sentencepiece模块，训练bpe的tokenizer
"""
import copy

import jieba
from collections import defaultdict


class PyBPE():
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = defaultdict(int)
        self.merges = {}

    def __process(self, text):
        words = text.split()
        vocab = []
        for word in words:
            word = f"<{word}>"
            vocab.append(list(word))
        return vocab

    def __init_corpus(self, words):
        """
        初始化：从每个单词的字符级分割开始，初始词汇表包含所有可能的字符
        Returns:
        """
        for word in words:
            for char in word:
                self.vocab[char] += 1

    def __get_stats(self, words):
        """
        统计频率：对语料中所有字符对的出现频率进行统计
        """
        pairs = defaultdict(int)
        for word in words:
            symbols = word
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += 1
        return pairs

    def __merge_vocab(self, pair, words):
        """
        合并操作：找到最频繁的字符对，将其合并为一个新的子词，并更新词汇表
        Returns:

        """
        replace_word = ''.join(list(pair))
        replace_target = ' '.join(list(pair))

        new_words = []
        for word in words:
            word = ' '.join(word)
            if replace_target in word:
                word = word.replace(replace_target, replace_word)
            new_words.append(word.split(' '))
        return new_words

    def train(self, corpus):
        """
        bpe分词算法
        """
        # 初始化词汇表（字符级）
        words = self.__process(corpus)
        self.__init_corpus(words)


        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            # 获取字符对
            pairs = self.__get_stats(words)
            if not pairs:
                break
            # 每次选择一个出现频率最大的进行合并
            best_pair = max(pairs, key=pairs.get)
            words = self.__merge_vocab(best_pair, words)

            # 记录merge合并的顺序，越小，合并优先级越高
            self.merges[best_pair] = i

            # 将新合并的符号加入词汇表
            merge_word = ''.join(best_pair)
            self.vocab[merge_word] += pairs[best_pair]
        return self.merges

    def encode(self, text):
        """
        编码，同样，拆分成字符级，然后不断地进行合并，得到最优拆分粒度
        Args:
            text: 编码文本

        Returns:
            分词结果
        """

        words = self.__process(text)
        tokens = []
        for word in words:
            while len(word) > 1:
                pairs = self.__get_stats([word])
                if not pairs:
                    break

                best_pair = min(pairs.keys(), key=lambda pair: self.merges.get(pair, float('inf')))

                if best_pair not in self.merges:
                    break
                word = self.__merge_vocab(best_pair, [word])[0]
            tokens.extend(word)
        return tokens

    def decode(self, tokens):
        """
        解码，还原文本，根据单词起始、终止边界来进行还原
        Args:
            tokens: 分词结果

        Returns:
            还原文本
        """
        text = ''.join(tokens)
        text = text.replace('><', ' ').replace('<', '').replace('>', ' ')
        return text


if __name__ == '__main__':
    # BPE只是一种分词算法，需要进一步映射词表，处理oov，才可以用于大模型训练
    # corpus = "hello world, how are you? nice to meet you, good morning, do you have a work?"
    corpus = "apple apply ape banana"
    vocab_max_size = 50
    bpe = PyBPE()
    merges = bpe.train(corpus)
    text = "app aple anban abp ly ana apple banan"
    tokens = bpe.encode(text)
    print(tokens)
    print(bpe.decode(tokens))