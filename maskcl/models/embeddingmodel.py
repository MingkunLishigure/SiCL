from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

from einops import rearrange, repeat
from torch.nn import init
import pdb

class Embedding_model(nn.Module):
    def __init__(self, d_model):
        super(Embedding_model,self).__init__()

        self.num_features = d_model
        self.full_connect = nn.Linear(self.num_features * 2, self.num_features)
        init.kaiming_normal_(self.full_connect.weight, mode='fan_out')
        init.constant_(self.full_connect.bias, 0)
        
        self.full_bn = nn.BatchNorm1d(self.num_features)
        self.full_bn.bias.requires_grad_(False)
        init.kaiming_normal_(self.full_connect.weight, mode='fan_out')
        init.constant_(self.full_bn.bias, 0)

    def forward(self, features1, features2):
        
        
        features3 = torch.cat([features1.clone().detach(),features2.clone().detach()], dim = 1)
        features3 = self.full_connect(features3)
        features3 = self.full_bn(features3)
        bn_x = F.normalize(features3)
        
        
        if self.training == False:    
            return bn_x
        else:
            return bn_x
        
    