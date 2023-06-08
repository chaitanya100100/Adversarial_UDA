import torch
from models.domain_adversarial_network import GradientReverseLayer
import torch.nn.functional as F
import numpy as np

def get_bsp_loss(x, k=1):
    if x.shape[0] == 0:
        return torch.tensor(0, device=x.device, dtype=x.dtype)
    singular_values = torch.linalg.svdvals(x)
    return torch.sum(singular_values[:k] ** 2)


class NuclearWassersteinDiscrepancy(torch.nn.Module):
    def __init__(self, classifier: torch.nn.Module):
        super(NuclearWassersteinDiscrepancy, self).__init__()
        self.grl = GradientReverseLayer()
        self.classifier = classifier

    @staticmethod
    def n_discrepancy(y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        pre_s, pre_t = F.softmax(y_s, dim=1), F.softmax(y_t, dim=1)
        loss = -torch.norm(pre_t, 'nuc') / y_t.shape[0] + torch.norm(pre_s, 'nuc') / y_s.shape[0]
        return loss

    def forward(self, f, num_s):
        f_grl = self.grl(f)
        y = self.classifier(f_grl)
        y_s, y_t = y[:num_s], y[num_s:]
        loss = self.n_discrepancy(y_s, y_t)
        return loss

def get_bnm_loss(x):
    x = torch.nn.functional.softmax(x, dim=-1)
    return -torch.linalg.norm(x, "nuc") / x.shape[0]

def entropy(x):
    bs = x.size(0)
    epsilon = 1e-5
    entropy = -x * torch.log(x + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy 
