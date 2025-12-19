import itertools
import os
from functools import partial
from typing import BinaryIO
from multiprocessing import pool, Pool
import regex as re

from cs336_basics.train_tokenizer import train_tokenizer


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


def multi_process_pretokenizer(path, special_tokens: list[str]) -> list[str]:
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
    results = list(itertools.chain.from_iterable(results))
    return results