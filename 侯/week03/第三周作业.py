"""
多分类任务：
'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', '你', '我', '他'
’你‘字符是第几位，就是第几类
"""

import json
import random

import torch
import torch.nn as nn
import numpy as np
