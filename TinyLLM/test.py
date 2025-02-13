from TinyLLM.tokenizer import PyBPE


def pybpe_test():
    # corpus = "hello world, how are you? nice to meet you, good morning, do you have a work?"
    corpus = "在Python中，defaultdict 是一个非常有用的数据结构，它是collections模块中的一个类，主要用于创建一个默认值的字典。与普通的字典不同，defaultdict 允许你在访问不存在的键时返回一个默认值，而不是抛出KeyError"
    vocab_max_size = 50
    bpe = PyBPE()
    merges = bpe.train(corpus)
    text = "defaultdict 在python中是用来干嘛的"
    tokens = bpe.encode(text)
    print(tokens)
    print(bpe.decode(tokens))


if __name__ == '__main__':
    pybpe_test()