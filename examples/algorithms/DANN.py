from typing import Dict, List

import torch

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.domain_adversarial_network import DomainAdversarialNetwork
from models.initializer import initialize_model
from optimizer import initialize_optimizer_with_model_params
from losses import initialize_loss
from utils import concat_input
from models.cp_impl import get_bsp_loss, NuclearWassersteinDiscrepancy, entropy


class DANN(SingleModelAlgorithm):
    """
    Domain-adversarial training of neural networks.

    Original paper:
        @inproceedings{dann,
          title={Domain-Adversarial Training of Neural Networks},
          author={Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand and Lempitsky},
          booktitle={Journal of Machine Learning Research 17},
          year={2016}
        }
    """

    def __init__(
        self,
        config,
        d_out,
        grouper,
        loss,
        metric,
        n_train_steps,
        n_domains,
        group_ids_to_domains,
    ):
        # Initialize model
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        self.dann_type = config.dann_type
        model = DomainAdversarialNetwork(featurizer, classifier, n_domains, self.dann_type, num_classes=d_out)
        parameters_to_optimize: List[Dict] = model.get_parameters_with_lr(
            featurizer_lr=config.dann_featurizer_lr,
            classifier_lr=config.dann_classifier_lr,
            discriminator_lr=config.dann_discriminator_lr,
        )
        self.optimizer = initialize_optimizer_with_model_params(config, parameters_to_optimize)
        self.domain_loss = initialize_loss('cross_entropy', config)

        # Initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.group_ids_to_domains = group_ids_to_domains

        # Algorithm hyperparameters
        self.penalty_weight = config.dann_penalty_weight

        # Additional logging
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("domain_classification_loss")

        print("DANN Type is ", self.dann_type)
        self.use_bsp = config.use_bsp_loss
        if self.use_bsp:
            self.logged_fields.append("bsp_loss")
            print("Using BSP loss...")
        
        self.use_nwd = config.use_nwd_loss
        self.nwd_loss_weight = config.nwd_loss_weight
        self.avoid_dann = config.avoid_dann
        if self.use_nwd:
            self.logged_fields.append("nwd_loss")
            print("Using NWD loss...")
            self.nwd_loss_module = NuclearWassersteinDiscrepancy(classifier)
        if self.avoid_dann:
            print("Avoiding DANN...")

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch 
                - domains_true (Tensor): true domains for batch and unlabeled batch
                - domains_pred (Tensor): predicted domains for batch and unlabeled batch
                - unlabeled_features (Tensor): featurizer outputs for unlabeled_batch
        """
        # Forward pass
        x, y_true, metadata = batch
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        domains_true = self.group_ids_to_domains[g]

        if unlabeled_batch is not None:
            unlabeled_x, unlabeled_metadata = unlabeled_batch
            unlabeled_domains_true = self.group_ids_to_domains[
                self.grouper.metadata_to_group(unlabeled_metadata)
            ]

            # Concatenate examples and true domains
            x_cat = concat_input(x, unlabeled_x)
            domains_true = torch.cat([domains_true, unlabeled_domains_true])
        else:
            x_cat = x
            
        x_cat = x_cat.to(self.device)
        y_true = y_true.to(self.device)
        domains_true = domains_true.to(self.device)
        y_pred, domains_pred, features = self.model(x_cat)

        bsp_loss = 0
        if self.use_bsp:
            bsp_loss = get_bsp_loss(features[: len(y_true)])
            if features.shape[0] > len(y_true):
                bsp_loss += get_bsp_loss(features[len(y_true): ])

        nwd_loss = 0
        if self.use_nwd and unlabeled_batch is not None:
            nwd_loss = -self.nwd_loss_module(features, len(y_true))

        # Ignore the predicted labels for the unlabeled data
        y_pred_full = y_pred.clone()
        y_pred = y_pred[: len(y_true)]

        return {
            "g": g,
            "metadata": metadata,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_full": y_pred_full,
            "domains_true": domains_true,
            "domains_pred": domains_pred,
            "bsp_loss": bsp_loss,
            "nwd_loss": nwd_loss,
            'features': features,
        }

    def objective(self, results):
        classification_loss = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )

        if self.is_training and not self.avoid_dann:
            if self.dann_type in ['dann', 'cdan']:
                domain_classification_loss = self.domain_loss.compute(
                    results.pop("domains_pred"),
                    results.pop("domains_true"),
                    return_dict=False,
                )
            else:
                y_prob_full = torch.nn.functional.softmax(results.pop('y_pred_full'), -1).detach()
                weight = 1.0 + torch.exp(-entropy(y_prob_full))
                weight = weight / torch.sum(weight) * y_prob_full.shape[0]
                # dp = results.pop('domains_pred')
                # domain_classification_loss = torch.nn.functional.binary_cross_entropy(
                #     dp[..., 1]-dp[..., 0],
                #     results.pop('domains_true').float(),
                #     weight=weight,
                # )
                domain_classification_loss = torch.nn.functional.cross_entropy(
                    results.pop('domains_pred'),
                    results.pop('domains_true'),
                    reduction='none'
                )
                domain_classification_loss = (domain_classification_loss * weight).mean()
        else:
            domain_classification_loss = 0.0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(
            results, "domain_classification_loss", domain_classification_loss
        )
        final_loss = classification_loss + domain_classification_loss * self.penalty_weight
        
        if self.use_bsp:
            bsp_loss = results.pop('bsp_loss')
            final_loss += bsp_loss * 1.e-4
            self.save_metric_for_logging(results, "bsp_loss", bsp_loss)
        
        if self.use_nwd:
            nwd_loss = results.pop('nwd_loss')
            final_loss += nwd_loss * self.nwd_loss_weight
            self.save_metric_for_logging(results, "nwd_loss", nwd_loss)
        return final_loss