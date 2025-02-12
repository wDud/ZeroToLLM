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
    def __init__(self, corpus, vocab_max_size):
        self.corpus = corpus
        self.vocab_max_size = vocab_max_size
        self.vocab = defaultdict(int)
        self.pairs = defaultdict(int)

    def __init_corpus(self, mode='spacewhite_split'):
        """
        初始化：从每个单词的字符级分割开始，初始词汇表包含所有可能的字符
        Args:
            mode: 分割方式，英文使用空格，中文使用jieba分词

        Returns:

        """
        if mode == 'spacewhite_split':
            words = self.corpus.split()
        else:
            words = jieba.lcut(self.corpus)
        for word in words:
            word = f"<{word}>"
            symbols = list(word)
            key = ' '.join(symbols)
            self.vocab[key] += 1

    def __stistic_freq(self):
        """
        统计频率：对语料中所有字符对的出现频率进行统计
        """
        pairs = defaultdict(int)
        for word in self.vocab:
            symbols = word.split()
            if len(symbols) == 1:
                pairs[symbols[0]] += 1
                continue
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += 1

        self.pairs = pairs


    def __merge_pair(self):
        """
        合并操作：找到最频繁的字符对，将其合并为一个新的子词，并更新词汇表
        Returns:

        """
        while True:
            high_freq_pair = max(self.pairs, key=self.pairs.get)
            if self.pairs[high_freq_pair] == 1:
                break
            replace_word = high_freq_pair[0] + high_freq_pair[1]
            replace_target = " ".join(list(high_freq_pair))
            vocab = defaultdict(int)
            pairs = copy.deepcopy(self.pairs)
            for word in self.vocab:
                if replace_target in word:
                    word = word.replace(replace_target, replace_word)
                vocab[word] += 1
            self.vocab = vocab
            self.__stistic_freq()

    def bpe_segment(self, num_merges=5):
        """
        bpe分词算法
        """
        self.__init_corpus()
        self.__stistic_freq()
        self.__merge_pair()

        vocab = []
        for symbol in self.vocab:
            words = symbol[1: -1].split(" ")
            vocab.extend(words)
        self.vocab = [_ for _ in set(vocab) if _]



if __name__ == '__main__':
    # corpus = "hello world, how are you? nice to meet you, good morning, do you have a work?"
    corpus = "hug pug pun bun hugs"
    vocab_max_size = 50
    bpe = PyBPE(corpus, vocab_max_size)
    bpe.bpe_segment()
    print(bpe.vocab)