from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


class DomainDiscriminator(nn.Sequential):
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_
    In the original paper and implementation, we distinguish whether the input features come
    from the source domain or the target domain.

    We extended this to work with multiple domains, which is controlled by the n_domains
    argument.

    Args:
        in_feature (int): dimension of the input feature
        n_domains (int): number of domains to discriminate
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, n_domains)`
    """

    def __init__(
        self, in_feature: int, n_domains, hidden_size: int = 1024, batch_norm=True
    ):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_domains),
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, n_domains),
            )

    def get_parameters_with_lr(self, lr) -> List[Dict]:
        return [{"params": self.parameters(), "lr": lr}]

class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class RandomizedMultiLinearMap(torch.nn.Module):

    def __init__(self, feat_dim, num_classes, map_dim = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.scaler = np.sqrt(float(map_dim))
        Rf = torch.randn(feat_dim, map_dim)
        Rg = torch.randn(num_classes, map_dim)
        self.register_buffer('Rf', Rf, persistent=True)
        self.register_buffer('Rg', Rg, persistent=True)
        self.output_dim = map_dim

    def forward(self, f, g):
        f = f @ self.Rf
        g = g @ self.Rg
        output = (f * g) / self.scaler
        return output


class DomainAdversarialNetwork(nn.Module):
    def __init__(self, featurizer, classifier, n_domains, dann_type, num_classes=None, cdan_map_dim=1024):
        super().__init__()
        self.dann_type = dann_type
        assert self.dann_type in ['dann', 'cdan', 'cdane']
        if self.dann_type == 'cdane':
            assert n_domains == 2
        
        self.featurizer = featurizer
        self.classifier = classifier
        if self.dann_type == 'dann':
            self.domain_classifier = DomainDiscriminator(featurizer.d_out, n_domains)
        else:
            self.domain_classifier = DomainDiscriminator(cdan_map_dim, n_domains)

        self.gradient_reverse_layer = GradientReverseLayer()

        if self.dann_type in ['cdan', 'cdane']:
            self.multilinear_map = RandomizedMultiLinearMap(featurizer.d_out, num_classes, map_dim=cdan_map_dim)

    def forward(self, input):
        features = self.featurizer(input)
        y_pred = self.classifier(features)
        ret_features = features.clone()
        if self.dann_type == 'dann':
            features = self.gradient_reverse_layer(features)
            domains_pred = self.domain_classifier(features)
        else:
            y_prob = torch.nn.functional.softmax(y_pred, -1).detach()
            dd_inp = self.multilinear_map(features, y_prob)
            dd_inp = self.gradient_reverse_layer(dd_inp)
            domains_pred = self.domain_classifier(dd_inp)
        return y_pred, domains_pred, ret_features

    def get_parameters_with_lr(self, featurizer_lr, classifier_lr, discriminator_lr) -> List[Dict]:
        """
        Adapted from https://github.com/thuml/Transfer-Learning-Library

        A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        # In TLL's implementation, the learning rate of this classifier is set 10 times to that of the
        # feature extractor for better accuracy by default. For our implementation, we allow the learning
        # rates to be passed in separately for featurizer and classifier.
        params = [
            {"params": self.featurizer.parameters(), "lr": featurizer_lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr},
        ]
        return params + self.domain_classifier.get_parameters_with_lr(discriminator_lr)
