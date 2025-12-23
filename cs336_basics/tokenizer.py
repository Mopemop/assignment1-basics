import itertools
from collections import Counter
from collections.abc import Iterable, Iterator
from urllib.parse import to_bytes

import regex as re

from cs336_basics.train_tokenizer import multi_process_pretokenizer, pretokenizer, NodeList


class Tokenizer:
    def __init__(self,vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = None

    # 从序列化词汇和合并列表（格式与你的 BPE 训练代码输出相同）以及（可选）特殊令牌列表中构造并返回分词器
    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        texts: list[str] = []
        if self.special_tokens is not None:
            texts = re.split(f"({"|".join(re.escape(s) for s in self.special_tokens)})", text)
        else:
            texts.append(text)
        # 将str转为bytes
        split_texts: list[str] = []
        for text in texts:
            if self.special_tokens is not None and text not in self.special_tokens:
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                split_texts.extend(re.findall(PAT, text))
            else:
                split_texts.append(text)
        result: list[int] = []
        # 反转vocab查询bytes对应的int
        reverse_dict = {v: k for k, v in self.vocab.items()}
        for t in split_texts:
            bytes_list: list[bytes] = []
            encoded = t.encode("utf-8")
            for b in encoded:
                bytes_list.append(bytes([b]))
            if self.special_tokens is not None and t in self.special_tokens:
                result.append(reverse_dict[encoded])
                continue
            while True:
                length = len(bytes_list)
                pair_set = set()
                for index in range(1, length):
                    # 建立pair_set
                    pair_set.add((bytes_list[index - 1], bytes_list[index]))
                # 寻找最先符合的merge
                new_list: list[bytes] = []
                choose_merge: tuple[bytes, bytes] = None
                for merge in self.merges:
                    if merge in pair_set:
                        choose_merge = merge
                        break
                if choose_merge is None:
                    break
                # 合并
                i = 0
                while i < length:
                    # 检查是否可以与下一个字节合并
                    if i + 1 < length and (bytes_list[i], bytes_list[i + 1]) == choose_merge:
                        new_list.append(choose_merge[0] + choose_merge[1])
                        i += 2  # 跳过两个
                    else:
                        new_list.append(bytes_list[i])
                        i += 1  # 只走一个
                bytes_list = new_list
            for b in bytes_list:
                result.append(reverse_dict[b])
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        results: list[int] = []
        for string in iterable:
            # if string is not None:
            results.extend(self.encode(string))
        return iter(results)

    def decode(self, ids: list[int]) -> str:
        tokens: bytes = b""
        for i in ids:
            tokens += self.vocab[i]
        return tokens.decode("utf-8", "replace")