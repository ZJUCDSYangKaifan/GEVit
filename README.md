## Group Equivariant Vision Transformer

This repository contains the source code accompanying the paper:

 [Group Equivariant Vision Transformer](https://openreview.net/forum?id=uVG_7x41bN),  UAI 2023.
 
 Code Author: [Kaifan Yang](https://github.com/CSK0x26B/) & [Ke Liu](https://github.com/zjuKeLiu)

#### Abstract
*Vision Transformer (ViT) has achieved remarkable performance in computer vision. However, positional encoding in ViT makes it substantially difficult to realize the equivariance, compared to models based on convolutional operations which are translation-equivariant. Initial attempts have been made on designing equivariant ViT but proved not effective in some cases in this paper. To address this issue, we propose a Group Equivariant Vision Transformer (GE-ViT) via a novel, effective positional encoding operation. We prove that GE-ViT meets all the theoretical requirements of an equivariant neural network. Comprehensive experiments are conducted on standard benchmark datasets. The empirical results demonstrate that GE-ViT has made significant improvement over non-equivariant self-attention networks*

### Reproducing experimental results

#### Command for running rot-MNIST
```
python run_experiment.py --config.dataset rotMNIST --config.model p4sa --config.norm_type LayerNorm --config.attention_type Local --config.activation_function Swish --config.patch_size 9 --config.dropout_att 0.1 --config.dropout_values 0.1 --config.whitening_scale 1.41421356 --config.epochs 300 --config.optimizer Adam --config.lr 0.001 --config.optimizer_momentum 0.9 --config.scheduler constants --config.sched_decay_steps='(1000,)' --config.sched_decay_factor 1.0 --config.weight_decay 0.0001 --config.batch_size 8 --config.device cuda --config.seed 0 --config.comment ''
```

#### Command for running  CIFAR-10
```
python run_experiment.py --dataset CIFAR10 --model mz2sa --norm_type LayerNorm --attention_type Local --activation_function Swish --patch_size 5 --dropout_att 0.1 --dropout_values 0.0 --whitening_scale 1.41421356 --epochs 350 --optimizer SGD --lr=0.01 --optimizer_momentum 0.9 --scheduler linear_warmup_cosine --optimizer_decay_steps 1000 --optimizer_decay_factor 1.0 --weight_decay 0.0001 --batch_size 24 --device cuda --seed 0 --comment ""
```

#### Command for running PatchCamelyon
```
python run_experiment.py --dataset PCam --model p4sa --norm_type LayerNorm --attention_type Local --activation_function Swish --patch_size 5 --dropout_att 0.1 --dropout_values 0.1 --whitening_scale 1.41421356 --epochs 100 --optimizer SGD --lr 0.01 --optimizer_momentum 0.9 --scheduler linear_warmup_cosine --optimizer_decay_steps 1000 --optimizer_decay_factor 1.0 --weight_decay 0.0001 --batch_size 16 --device cuda --seed 0 --comment ""
```

### Note
Our code was modified based on the code presented in paper A. We mainly modified the “construct_relative_positions” function of the g_selfatt/groups/SE2.py and g_selfatt/g_selfatt/groups/E2.py module in [GSA-Nets](https://github.com/dwromero/g_selfatt) which corresponds to the part of position encoding. 

From the experimental results, there are differences between the results in our paper and those in [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) and we suspect that this is caused by differences in the experimental environment. The paper of [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) uses NVIDIA TITAN RTX, while we used NVIDIA Tesla A100. To ensure a fair comparison, we re-ran the code of [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) on our hardware. 

The experimental results of rot-MNIST and PatchCamelyon are similar to those presented in the paper, but the results of CIFAR-10 differ significantly from the paper. It is worth mentioning that in the [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M), the authors mentioned that they did not use automatic mixed precision when conducting experiments on the CIFAR-10 datasets. However, when we tried to run the experiments without using automatic mixed precision, we found that at the beginning of the training, the loss would become 'nan', and not converge. When we used automatic mixed precision, the loss converged to a smaller value and the model achieved high accuracy in prediction. The results presented in our paper were obtained using automatic mixed precision. Therefore, the experimental results presented in the table may not be consistent with those reported in the original [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M) paper. The experimental logs can be found in the folder "CIFAR10_EXP_LOG".

### Contributions of each author to the paper

[Kaifan Yang](https://github.com/CSK0x26B) & [Ke Liu](https://github.com/zjuKeLiu) proposed the innovative ideas, designed the model architecture and completed the initial draft of the paper. Fengxiang He made detailed revisions to the paper and provided many constructive comments.

### Acknowledgements
*We gratefully acknowledge the authors of GSA-Nets paper David W. Romero and Jean-Baptiste Cordonnier.  They patiently answered and elaborated on the experimental details of the paper [GSA-Nets](https://openreview.net/forum?id=JkfYjnOEo6M).*
