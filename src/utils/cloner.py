import copy
import torch.nn as nn

def get_clones(module, n):
    module_list = []
    for _ in range(n):
        module_list.append(copy.deepcopy(module))
    return nn.ModuleList(module_list)