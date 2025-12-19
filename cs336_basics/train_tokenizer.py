

def train_tokenizer(bytes_list: list[tuple[list[bytes], int]], vocab_size: int, cur_size: int, vocab_dict: dict[int, bytes]) \
        -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    # 反复merge至词表长度到vocab_size或无法扩增
    while cur_size < vocab_size:
        # 从所有bytes_list中找到最多的pair
        new_pair: dict[tuple[bytes, bytes], int] = {}
        for b_list, count in bytes_list:
            length = len(b_list)
            for index in range(1, length):
                key = (b_list[index-1], b_list[index])
                new_pair[key] = new_pair.get(key, 0) + count
                index += 1
        if new_pair == {}:
            break
        best_pair = max(new_pair, key=lambda k: (new_pair[k], k))
        # 将pair记录进merges和vocab_dict
        merges.append(best_pair)
        merged_bytes = best_pair[0] + best_pair[1]
        vocab_dict[cur_size] = merged_bytes
        cur_size += 1
        # 更新bytes_list融合上一个pair
        for b_list in bytes_list:
            length = len(b_list)
            index = 1
            while index < length:
                if b_list[index-1] == best_pair[0] and b_list[index] == best_pair[1]:
                    b_list[index-1:index+1] = [merged_bytes]
                    length -= 1
                    continue
                index += 1
    return vocab_dict, merges