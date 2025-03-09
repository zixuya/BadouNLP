# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        class_num = config["class_num"]
        print("class_num", class_num)
        num_layers = config["num_layers"]

        self.use_bert = (config["model_type"] == "bert")
        if self.use_bert:
            # self.layer = BertModel.from_pretrained(
            #     config["bert_model"], 
            #     num_hidden_layers=num_layers,
            #     return_dict=False)
            self.layer = BertForTokenClassification.from_pretrained(
                config["bert_model"], 
                num_labels=class_num,
                return_dict=False)
            hidden_size = self.layer.config.hidden_size # use hidden size from bert model
            self.classify = nn.Linear(hidden_size, class_num) # bert output is mapped to hidden_size, not 2 * hidden_size
        else:
            vocab_size = config["vocab_size"] + 1
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0) # not needed for bert
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
            self.classify = nn.Linear(hidden_size * 2, class_num) # bidirectional LSTM
                        
        if config["use_lora"]:
            self.lora_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                inference_mode=False,
                r=8,
                lora_alpha=16,              
                target_modules=["query", "key", "value"],           
                modules_to_save=["classifier"]       
            )
            self.layer = get_peft_model(self.layer, self.lora_config)
            print(f"Model converted to LoRA. Trainable parameters: {self.layer.print_trainable_parameters()}")

        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:
            x = self.layer(x)[0] #output shape:(batch_size, sen_len, hidden_size)
        else: # LSTM
            x = self.embedding(x)  #input shape:(batch_size, sen_len)
            x, _ = self.layer(x)  #input shape:(batch_size, sen_len, input_dim)
            
        predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if config.get("use_lora", False) and config["model_type"] == "bert":
        # Get only trainable parameters (LoRA params) to optimize
        trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
        if optimizer == "adam":
            return Adam(trainable_params, lr=learning_rate)
        elif optimizer == "sgd":
            return torch.optim.SGD(trainable_params, lr=learning_rate)
    else:
        # Regular optimization (all parameters)
        if optimizer == "adam":
            return Adam(model.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            return torch.optim.SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)