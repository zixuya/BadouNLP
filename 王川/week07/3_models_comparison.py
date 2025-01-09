# model_type    max_length      hidden_size     batch_size      lr      预测时间ms/100条data     准确率

# bert              25              256             128         1e-5        2.40                0.8949
# bert              25              256             128         1e-4        2.39                0.8816
# bert              25              256             128         1e-3        2.29                0.3336

# lstm              25              256             128         1e-5        0.89                0.8565
# lstm              25              256             128         1e-4        0.88                0.8774
# lstm              25              256             128         1e-3        0.88                0.8874

# grated_cnn        25              256             128         1e-5        0.85                0.8532(kernel_size = 3)
# grated_cnn        25              256             128         1e-4        0.85                0.8824(kernel_size = 3)
# grated_cnn        25              256             128         1e-3        0.84                0.8612(kernel_size = 3)
