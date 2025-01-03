#计算Bert参数量

sen_len = 4 #句子长度
hidden_size = 768 #向量维度
num_layers = 1 #隐藏层层数
intermediate_weight_size = 3072 #feed_forward层维度

#embedding层
token_enbedding = sen_len * hidden_size
Segment_embedding = 2 * hidden_size
Position_embedding = sen_len * hidden_size
layer_norm = hidden_size + hidden_size
embedding_total = token_enbedding + Segment_embedding + Position_embedding + layer_norm

#self_attention
q = hidden_size * hidden_size
k = hidden_size * hidden_size
v = hidden_size * hidden_size
self_attention_output = hidden_size * hidden_size + hidden_size
self_attention_layer_norm = hidden_size + hidden_size
self_attention_total = q + k + v + self_attention_output + self_attention_layer_norm

#feed_forward
feed_forward = hidden_size * intermediate_weight_size + intermediate_weight_size
feed_forward_output = hidden_size * intermediate_weight_size + hidden_size
self_attention_layer_norm = hidden_size + hidden_size
feed_forward_total = feed_forward + feed_forward_output + self_attention_layer_norm

single_transformer_layer = self_attention_total + feed_forward_total

total = embedding_total + num_layers * single_transformer_layer
print(total)
