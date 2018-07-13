# SwitchableNormalization-Keras
the **switchable normalization** method by keras

## Switchable Normalization
- the paper [Differentiable Learning-to-Normalize via Switchable Normalization](https://arxiv.org/abs/1806.10779) introduce the Switchable Normalization(SN).
- the blog [深度剖析 | 可微分学习的自适配归一化](https://zhuanlan.zhihu.com/p/39296570) explain SN method.
- the SN adapt to weight 3 different normalization method, including IN/LN/BN. It need 3 trainable mean weights and 3 trainable variance weight to weighted average means and variances respectively.So the trainable parameters of SN equal to **2*channel+6**.

## Code of Switchable Normalization
- [SN by Pytorch](https://github.com/switchablenorms/Switchable-Normalization/blob/master/models/switchable_norm.py)
- [SN by Keras in this repository](https://github.com/Bingohong/SwitchableNormalization-Keras)

## Experiment
- the detail of experiment log is in directory [experiments](https://github.com/Bingohong/SwitchableNormalization-Keras/experiments)
- compare 3 normalization method: **batch_norm(bn)** **group_norm(gn)** **switchable_norm(sn)**
- data: ISBI 2D EM segmentation images
- network: Unet(based on VGG)
- epoch: 5
- environment:GeForce 1080Ti
- consequent:
  - training time: sn > gn > bn
  - segmentation result: sn > gn > bn