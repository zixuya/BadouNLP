# 模型名称        位置编码	transformer结构	多头机制	    ff层设计	    归一化层选择	            激活函数	    是否使用bias
# baichuan2-7b	RoPE	    串行	            MHA	    gated形式	  RMSnorm/pre norm	          SiLU	      无bias
# baichuan2-13b	Alibi	    串行	            MHA	    gated形式	  RMSnorm/pre norm	          SiLU	      无bias
# chatglm2	    RoPE	    串行	            GQA	    gated形式	  RMSnorm/pre norm	          SiLU	      qkv有bias，其他线性层无bias
# chatglm3	    RoPE	    串行	            GQA	    gated形式	  RMSnorm/pre norm	          SiLU	      qkv有bias，其他线性层无bias
# DBRX	        RoPE	    串行	            GQA	    MoE	        LayerNorm/sandwich norm	    动态	      无bias
# gemma	        RoPE	    串行	            MHA	    gated形式	  RMSnorm/pre norm	          GeLU	      无bias
# grok1	        RoPE	    串行	            MHA	    MoE	        RMSnorm/sandwich norm	      GeLU	      无bias
# llama	        RoPE	    串行	            MHA	    gated形式	  RMSnorm/pre norm	          SiLU	      无bias
# llama2	      RoPE	    串行	            GQA	    gated形式	  RMSnorm/pre norm	          SiLU	      无bias
# Mixtral	      RoPE	    串行	            GQA	    MoE	        RMSnorm/pre norm	          SiLU	      无bias
# moss	        RoPE	    平行	            MHA	    传统方式	    LayerNorm	                  gelu_new    sa无bias，ff有bias
# qwen	        RoPE	    串行	            MHA	    gated形式	  RMSnorm/pre norm	          SiLU	      qkv有bias，其他线性层无bias
# deepseek	    RoPE	    串行	            MLA	    MoE	        RMSnorm/pre norm	          SiLU	      无bias
