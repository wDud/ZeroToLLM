"""
simple llm achieve
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataclasses import dataclass

torch.manual_seed(1024)


@dataclass
class GPTConfig:
    max_seq_len: int = 512   # 这里其实应该是文本的最大长度（ max_seq_len）
    block_num: int = 12
    n_layer: int = 6
    n_head: int = 12
    hidden_size: int = 768
    head_size: int = hidden_size // n_head
    vocab_size: int = 50257
















