import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchtext

import numpy as np
import math
import os

import nltk
import re
import spacy
import pandas as pd

if __name__=="__main__":
    if (torch.cuda.is_available()):
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Use {device}")
    device = torch.device(device)