# Dual-Decoding Branch U-shaped Net (DBUNet)
paper address: https://www.sciencedirect.com/science/article/abs/pii/S1047320323001062


Dual-Decoding Branch U-shaped Net
1) Combining Transformer ViT structure with DECODER part to achieve feature fusion. Considering the asymmetry of semantic feature information of codec, ViT Encoder was embedded into the Decoder part to realize feature strengthening.
2) A new Attention Mechanism of Difference Amplification is proposed. We embedded a kind of polarized attention module in the ViT that enhanced or suppressed the weight of different channels.
3) A new Feature Fusion Bottleneck is proposed to fully excavate and absorb the feature information. This module plays a key role in feature fusion.

# Env

IDE:	Pycharm 2020.1 Professional ED.

Language:	Python 3.8.15

Framework:	Pytorch 1.13.0

CUDA:	Version 11.7 

# Model
![image](https://github.com/YF-W/DBUNet/assets/66008255/36897af5-a9c9-4962-90b2-1332392f0cc3)

