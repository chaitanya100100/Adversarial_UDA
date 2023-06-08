## dann
python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/dann/ --dataset iwildcam --algorithm DANN --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.00031798006858157785 --dann_featurizer_lr 3.179800685815778e-05 --dann_discriminator_lr 0.00010569049202386628 --dann_penalty_weight 0.39361280014637084 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True

--eval_only --eval_save_data --progress_bar

## dann_bsp
python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/dann_bsp/ --dataset iwildcam --algorithm DANN --use_bsp_loss True --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.00031798006858157785 --dann_featurizer_lr 3.179800685815778e-05 --dann_discriminator_lr 0.00010569049202386628 --dann_penalty_weight 0.39361280014637084 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True

python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/dann_bsp_randaug/ --dataset iwildcam --algorithm DANN --use_bsp_loss True --additional_train_transform randaugment --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.00031798006858157785 --dann_featurizer_lr 3.179800685815778e-05 --dann_discriminator_lr 0.00010569049202386628 --dann_penalty_weight 0.39361280014637084 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True


## dann_nwd
python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/dann_nwd/ --dataset iwildcam --algorithm DANN --use_nwd_loss True --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.00031798006858157785 --dann_featurizer_lr 3.179800685815778e-05 --dann_discriminator_lr 0.00010569049202386628 --dann_penalty_weight 0.39361280014637084 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True

## dann_nwd_avoiddann (daln -discriminator free adv learning network)
python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/dann_nwd_avoiddann/ --dataset iwildcam --algorithm DANN --use_nwd_loss True --avoid_dann True --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.0003 --dann_featurizer_lr 3.2e-5 --dann_discriminator_lr 0.0001 --dann_penalty_weight 0.4 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True

python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/dann_nwd_avoiddann_w1/ --dataset iwildcam --algorithm DANN --use_nwd_loss True --avoid_dann True --nwd_loss_weight 1. --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.0003 --dann_featurizer_lr 3.2e-5 --dann_discriminator_lr 0.0001 --dann_penalty_weight 0.4 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True

python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/dann_nwd_avoiddann_wneg1/ --dataset iwildcam --algorithm DANN --use_nwd_loss True --avoid_dann True --nwd_loss_weight=-1. --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.0003 --dann_featurizer_lr 3.2e-5 --dann_discriminator_lr 0.0001 --dann_penalty_weight 0.4 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True

python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/dann_nwd_avoiddann_neg/ --dataset iwildcam --algorithm DANN --use_nwd_loss True --avoid_dann True --nwd_loss_weight=-0.1 --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.0003 --dann_featurizer_lr 3.2e-5 --dann_discriminator_lr 0.0001 --dann_penalty_weight 0.4 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True


## cdan
python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/cdan/ --dataset iwildcam --algorithm DANN --dann_type cdan --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.00031798006858157785 --dann_featurizer_lr 3.179800685815778e-05 --dann_discriminator_lr 0.00010569049202386628 --dann_penalty_weight 0.39361280014637084 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True

python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/cdane/ --dataset iwildcam --algorithm DANN --dann_type cdane --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --dann_classifier_lr 0.00031798006858157785 --dann_featurizer_lr 3.179800685815778e-05 --dann_discriminator_lr 0.00010569049202386628 --dann_penalty_weight 0.39361280014637084 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 6 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True


## ermnoaug_bnm
python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/ermnoaug_bnm --dataset iwildcam --algorithm ERM --use_bnm_loss True --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --lr 3.4e-05 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 12 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True 

python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/ermnoaug_bnm_wneg1 --dataset iwildcam --algorithm ERM --use_bnm_loss True --bnm_loss_weight=-1. --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --lr 3.4e-05 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 12 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True 


python examples/run_expt.py --log_dir /vision/u/chpatel/dstest/erm_orig --dataset iwildcam --algorithm ERM --seed 0 --unlabeled_split extra_unlabeled --groupby_fields from_source_domain --gradient_accumulation_steps 4 --lr 3.4e-05 --unlabeled_batch_size 18 --batch_size 6 --n_epochs 12 --loader_kwargs num_workers=4 pin_memory=True --unlabeled_loader_kwargs num_workers=8 pin_memory=True --eval_only --eval_save_data --progress_bar


## Some observations
- ERN results are with randaugment and DANN CORAL results are without randaugment, even though the paper claims otherwise.
