import itertools
import os
from collections import Counter
from functools import partial
from typing import BinaryIO
from multiprocessing import pool, Pool
import regex as re


class Node:
    def __init__(self, data=None):
        self.data = data
        self.prev = None
        self.next = None

    def __repr__(self):
        return str(self.data)

class NodeList:
    def __init__(self):
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.length = 0

    def append(self, data):
        node = Node()
        node.data = data
        node.next = self.tail
        node.prev = self.tail.prev
        self.tail.prev = node
        node.prev.next = node
        self.length += 1

    def prepend(self, data):
        node = Node()
        node.data = data
        node.prev = self.head
        node.next = self.head.next
        self.head.next = node
        node.next.prev = node
        self.length += 1

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenizer(chunk, special_tokens) -> list[str]:
    # 先针对special_tokens进行转义加分割
    chunk_without_special = re.split("|".join(re.escape(s) for s in special_tokens), chunk)
    # 预分词的分割正则表达式
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    result: list[str] = []
    for string in chunk_without_special:
        result.extend(re.findall(PAT, string))
    return result


def multi_process_pretokenizer(path, special_tokens: list[str]):
    ## Usage
    with open(path, "rb") as f:
        num_processes = 4
        # 分块
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        chunks: list[str] = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # 将\r \n两种分行形式统一为\n
            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunks.append(chunk)
    # 并行预分词
    worker_func = partial(pretokenizer, special_tokens=special_tokens)
    with Pool(num_processes) as p:
        results = p.map(worker_func, chunks)
    # 并为一维string数组
    text_preTokenized = list(itertools.chain.from_iterable(results))
    # 对字符串用hash表记录每个串出现次数
    string_counter = Counter(text_preTokenized)
    # 将str转为bytes
    bytes_list: list[tuple[NodeList, int]] = []
    for text, count in string_counter.items():
        encoded = text.encode("utf-8")
        new_node_list = NodeList()
        for b in encoded:
            # 采用bytes(b)会导致生成b大小数量的二进制字节，而不是b的bytes码(前者用来分配内存)
            new_node_list.append(bytes([b]))
        bytes_list.append((new_node_list, count))
    return bytes_list

def train_tokenizer(bytes_list: list[tuple[NodeList, int]], vocab_size: int, cur_size: int, vocab_dict: dict[int, bytes]) \
        -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    new_pair: dict[tuple[bytes, bytes], int] = {}
    # 从所有bytes_list中找到最多的pair
    for b_list, count in bytes_list:
        node = b_list.head.next
        tail = b_list.tail
        while node.next is not tail:
            key = (node.data, node.next.data)
            new_pair[key] = new_pair.get(key, 0) + count
            node = node.next
    # 反复merge至词表长度到vocab_size或无法扩增
    while cur_size < vocab_size:
        best_pair = max(new_pair, key=lambda k: (new_pair[k], k))
        # 将pair记录进merges和vocab_dict
        merges.append(best_pair)
        merged_bytes = best_pair[0] + best_pair[1]
        vocab_dict[cur_size] = merged_bytes
        cur_size += 1
        # 更新bytes_list融合上一个pair
        for b_list, count in bytes_list:
            node = b_list.head.next
            tail = b_list.tail
            head = b_list.head
            while node is not None and node.next is not None and node.next is not tail:
                if node.data == best_pair[0] and node.next.data == best_pair[1]:
                    # 更新new_pair
                    new_pair[best_pair] -= count
                    if node.next.next is not tail:
                        new_pair[(node.next.data, node.next.next.data)] -= count
                        new_pair[(merged_bytes, node.next.next.data)] = new_pair.get((merged_bytes, node.next.next.data), 0) + count
                    if node.prev is not head:
                        new_pair[(node.prev.data, node.data)] -= count
                        new_pair[(node.prev.data, merged_bytes)] = new_pair.get((node.prev.data, merged_bytes), 0) + count
                    node.data = merged_bytes
                    node.next.next.prev = node
                    node.next = node.next.next
                node = node.next
    return vocab_dict, merges