#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"
window_size = 3


def is_word_prefix(prefix, Dict):
    for key in Dict.keys():
        if key.startswith(prefix):
            return True
    return False

def all_cut(sentence, Dict):
    result = []
    
    # 如果句子为空，返回一个包含空列表的列表
    if not sentence:
        return [[]]
    
    # 从最大窗口开始，逐渐缩小窗口
    for i in range(min(window_size, len(sentence)), 0, -1):
        # 获取当前窗口的词
        current_word = sentence[:i]
        
        # 如果当前窗口的词在词典中
        if current_word in Dict:
            # 递归处理剩余的句子
            sub_results = all_cut(sentence[i:], Dict)
            
            # 将当前词插入到所有子结果的开头
            for sub_result in sub_results:
                result.append([current_word] + sub_result)
        elif i == 1 and is_word_prefix(current_word, Dict):
            sub_results = all_cut(sentence[i:], Dict)
            for sub_result in sub_results:
                result.append([current_word] + sub_result)
        elif len(sentence[i:]) == 1:
            result.append([current_word] + [sentence[i:]])
    return result

result = all_cut(sentence, Dict)
print('result', result)