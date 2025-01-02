# bert parameters
# 词表大小
v = 30522
# 维度
h = 768
#字段大小
max_len = 4
# 多头机制的头数为12
num_layers = 12
# embedding层
# token embeddings 输入层
token_embeddings = v * h
# segment embeddings 分段层
segment_embeddings = 2 * h
# position embeddings 位置层
position_embeddings = 512 * h
# LayerNorm 归一化层
#x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
#x = x * w + b
layernorm1 = 2*h*max_len

# transformer层
# self-Attention层
# 多头机制  查询（Q）、键（K）、值（V）的权重矩阵
# 先过线性层
liner1 = 3 * h * max_len
# num_layers * softmax(Q*K.swapaxes(1, 2)/np.sqrt(dk))*V
atten = num_layers * 3 * (h / num_layers) * max_len
# 再过一层线性层
liner2 = 3 * h * max_len
# bn层，使用残差机制，过归一化层
layernorm2 = 2*h*max_len
# Feed-Forward层,线性层H x 4H+激活函数+线性层4H x H
mlp = (max_len * 4*h + 4*h) + (4*max_len * h + h)
# bn层，使用残差机制，再过归一化层LayerNorm
layernorm3 = 2*h*max_len

all_parameters = token_embeddings + segment_embeddings + position_embeddings + layernorm1+ liner1 + atten + liner2 + layernorm2 + mlp + layernorm3

print(all_parameters)
