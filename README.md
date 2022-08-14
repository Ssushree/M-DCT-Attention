# M-DCT-Attention
This is the tensorflow implementation of "Multi-dimensional Discrete Cosine Transform based 3-D Attention Mechanism for Heterogeneous Periocular Recognition". 
This repository includes the proposed M-DCT attention module.

# The M-DCT attention module
The multi-dimensional DCT-based attention mechanism takes the convolutional feature map as input and generates 3-D attention coefficients at its output.
It performs multidimensional DCT transformation of the input feature map along the width, height, and channel dimensions (or modes).


![Multidimensional_DCT_Module_2](https://user-images.githubusercontent.com/35622430/184503604-7000ae3a-b627-4b4b-aa36-449c0cf62481.png)

# The M-DCT attention network
A 3-D attention mechanism embedded within a deep Siamese framework to match periocular images in heterogeneous wavelength range. The M-DCT attention network finds 
semantic similarities between cross-spectral periocular images and decides whether they belong to same or different classes.


![Tensor_DCT_Blockdiagram_2](https://user-images.githubusercontent.com/35622430/184503625-546e4319-e56b-4232-a6ae-dbac3e41d6f2.png)
