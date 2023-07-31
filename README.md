# Dual-Decoding Branch U-shaped Net (DBUNet)
Dual-Decoding Branch U-shaped Net
1) Combining Transformer ViT structure with DECODER part to achieve feature fusion. Considering the asymmetry of semantic feature information of codec, ViT Encoder was embedded into the Decoder part to realize feature strengthening.
2) A new Attention Mechanism of Difference Amplification is proposed. We embedded a kind of polarized attention module in the ViT that enhanced or suppressed the weight of different channels.
3) A new Feature Fusion Bottleneck is proposed to fully excavate and absorb the feature information. This module plays a key role in feature fusion.

# Env

IDE:	Pycharm 2020.1 Professional ED.

Language:	Python 3.8.15

Framework:	Pytorch 1.13.0

CUDA:	Version 11.7 


# FFB Testing Results (构造通用性特征融合组件)
1) Problem solving ideas
The question you mentioned is of great significance and will greatly contribute to the research quality of our manuscript. Following the previous question, we believe that the two recommendations are relevant and gradually deepening.

Through in-depth discussion, we believe that:
Other important open source models can be used to further test the effectiveness of our proposed modules.

The main methods are:
In the comparison method, select as many networks as possible, insert the modules we have constructed into them, and implement post-insertion testing in a completely consistent engineering environment to compare the network performance before and after the insertion.

The above experimental process not only achieves module verification, but also is an important part of our model tuning. We hope to apply this tuning method to as many other open-source models as possible to verify the tuning effect.


2) Specific solution to the problem
By analyzing the model of DBUNet and combining it with the profiling of all the comparative models, we have deployed the work as follows:

①	 The FFB Module will be transplanted for testing.
The FFB Module has the advantages of lightweight and pluggable, and at the same time is an important component of DBUNet proposed in this paper. Accordingly we plan to port the module for testing and analysis. On the other hand, in the discussion about AMDA, according to our observation, the Transformer structure in the comparison model does not adopt ViT architecture and AMDA has non-pluggable characteristics, so the AMDA module is not suitable for implementing comparisons.

②	 The test uses embedding rather than replacement.
In the early experiment of this modification, we used the REPLACEMENT method, but the effect was not satisfactory. We discussed that the original comparison network has a stable system with a close relationship before and after, and the modules are interconnected and strongly related. If imposed disassembly and replacement is used, it will have obvious damage to the original model, especially the absence of data processing module, which will greatly impair the analysis and image recovery ability of the network. Accordingly, we have switched to the addition of embedding to achieve the replacement of 10 models. It should be noted that in our comparison network, the DeepLab family accomplishes semantic segmentation through multiscale prediction, null convolution, and ASPP. However, these methods utilize cascaded structures, which make it difficult to embed portable modules.

These replaced models are: (a) U-Net; (b) SmaAt UNet; (c) Refine Net; (d) OAUNet; (e) Swin Transformer; (f) SUNet; (g) MTUNet; (h) FCN; (i) SegNet; (j) ResUNet

Their structure diagrams are:
![image](https://github.com/YF-W/DBUNet/assets/66008255/f44e40c3-b8f8-4b1c-9b2c-68b4169742c4)

3) Experimental results and analysis of the problem
![image](https://github.com/YF-W/DBUNet/assets/66008255/5047bd25-fd43-4414-906d-d3dfe6f90c2d)

The table above shows the percentage of performance differences when we embed the FFB Module into different comparison networks. We tested three datasets (All experimental conditions and parameters are completely consistent with previous experiments) to explore the effect of FFB, a channel fusion mechanism, on improving or reducing the performance of different modules. It should be noted that the values in the above table represent the "performance of the embedded module - performance of the original network (before embedding)" and are expressed as percentages. In addition, the green background color indicates that the FFB Module improves the original effect of the network.

We observe that the embedding effect of the FFB Module has both positive and negative results. On the positive side, the experimental data suggest that the FFB Module does have some model tuning effect on certain models in certain datasets, mainly due to the fact that the FFB achieves feature fusion in the most channel-rich regions, resulting in the elimination of channel segregation in the model. However, on the negative side, we found no specific regular pattern of model performance change. We tried to elaborate the reasons for this increase or decrease in terms of different types of networks (e.g., U-Shaped, Transformer Series), but individual data showing counterexamples did not support this conclusion. After careful analysis, we concluded that the original network is somewhat integral, and our deletion from the original base may cause damage to the overall structure. Although we designed pluggable modules, such modules may not be able to form a universal optimization enhancement to the existing mainstream and SOTA networks. 
