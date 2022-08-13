# M-DCT-Attention
This is the tensorflow implementation of "Multi-dimensional Discrete Cosine Transform based 3-D Attention Mechanism for Heterogeneous Periocular Recognition". 
This repository includes the proposed fine-grained dense module, committee of multi-feature attention module and simultaneous excitation module.

# The M-DCT attention module
The multi-dimensional DCT-based attention mechanism takes the convolutional feature map as input and generates 3-D attention coefficients at its output.
It performs multidimensional DCT transformation of the input feature map along the width, height, and channel dimensions (or modes).

# The M-DCT attention network
A 3-D attention mechanism embedded within a deep Siamese framework to match periocular images in heterogeneous wavelength range. The M-DCT attention network finds 
semantic similarities between cross-spectral periocular images and decides whether they belong to same or different classes.
