---
tags: ['deep-learning', 'playground']
---

# Playground/image-segmentation

This `TensorFlow` implementation of ENet is forked from @fregu856. You can find his 
original code here: [@fregu856/segmentation.git](https://github.com/fregu856/segmentation), which is implemented based on 
the pyTorch version at https://github.com/e-lab/ENet-training

### ENet Paper: 

The original paper for this model could be found here: 
[2016 Adam Paszke, et.al., ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
](https://arxiv.org/pdf/1606.02147.pdf)

## TODOs

- parallelize training to use more GPUs

## Common Problems

If you encounter the "No gradient defined for operation 'MaxPoolWithArgmax_1' (op type: MaxPoolWithArgmax)", you just need to use a GPU tensorflow instance, b/c the max_pool_with_argmax operator is not defined for the CPU.
