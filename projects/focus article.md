# MobileNetV2 (2018)
## 论文：
MobileNetV2: Inverted Residuals and Linear Bottlenecks
[论文链接](https://arxiv.org/abs/1801.04381)


## 文章背景与研究问题
**背景：**
Xception在ImageNet上实现了79％的top-1准确性和94.5％的top-5准确性，但是与以前的SOTA InceptionV3相比分别仅提高了0.8％和0.4％。新图像分类网络的边际收益越来越小，因此研究人员开始将注意力转移到其他领域。在资源受限的环境中，MobileNet推动了图像分类的重大发展。在移动和嵌入式设备上部署深度神经网络需要模型既小且高效，同时保持高性能。
**研究问题：**如何在不显著降低性能的情况下，减少卷积神经网络的参数数量和模型大小。
**主要方法或创新点**
MobileNet使用深度可分离卷积模块，并着重于高效和较少参数。可以针对不同的资源约束调整网络的规范方法。此外，它还总结了改进神经网络的最终解决方案：更大和更高的分辨率输入会导致更高的精度，更薄和更低的分辨率输入会导致更差的精度。通过引入倒置残差结构（Inverted Residuals）和线性瓶颈（Linear Bottlenecks），减少参数数量和模型大小。
1. **倒置残差结构（Inverted Residuals）**：传统残差网络（如ResNet）先压缩通道数再扩展，而MobileNetV2反其道而行——先通过**1x1卷积扩展通道数**，再使用**深度可分离卷积（Depthwise Separable Convolution）**提取空间特征，最后压缩通道。这种结构在减少计算量的同时保留了更多信息。  
2. **线性瓶颈层（Linear Bottleneck）**：在残差块的输出层**移除ReLU激活函数**，改用线性输出。因为低维空间中使用ReLU会破坏特征信息（如将负值归零），线性层能更完整地传递特征。  

通过**扩张因子（Expansion Ratio）**控制通道扩展程度，结合**宽度乘子（Width Multiplier）**调整模型整体大小，MobileNetV2在ImageNet数据集上仅用300万参数就达到72%的Top-1精度，比前代MobileNetV1显著提升，且计算量（FLOPs）降低15%。  
   
## 主要实验与结果
**数据：**
在ImageNet数据集上进行训练和测试，MobileNetV2在多个任务中表现出色，取得了优异的性能和效率。
**代码：**
[TensorFlow官方文档](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2#returns)
[官方github库](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
[官方example代码](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_example.ipynb)

## 应用场景、局限性与未来方向
**应用场景：**
广泛应用于图像分类、目标检测、语义分割等计算机视觉任务，尤其适用于移动和嵌入式设备。以及AR滤镜、无人机视觉导航等资源受限环境。
**局限性：**
尽管减少了参数数量和模型大小，但在极端高效的环境下仍然存在挑战。
**未来方向：**
1. 进一步优化倒置残差和线性瓶颈结构，探索更多的轻量级模型结构，提高模型在资源受限环境下的性能和效率。
2. “轻量化网络设计”，有多篇对比MobileNet系列与EfficientNet的深度分析


论文摘要：
在本文中，我们描述了一种新的移动架构 MobileNetV2，它提高了移动模型在多个任务和基准测试以及各种不同模型尺寸上的最先进性能。我们还描述了在我们称为 SSDLite 的新框架中将这些移动模型应用于对象检测的有效方法。此外，我们还演示了如何通过 DeepLabv3 的简化形式（我们称之为 Mobile DeepLabv3）构建移动语义分割模型。
MobileNetV2 架构基于倒置残差​​结构，其中残差块的输入和输出是薄瓶颈层，与使用输入中的扩展表示的传统残差模型相反，MobileNetV2 使用轻量级深度卷积来过滤中间扩展层中的特征。此外，我们发现，为了保持表示能力，消除窄层中的非线性非常重要。我们证明了这可以提高性能，并提供了导致这种设计的直觉。最后，我们的方法允许将输入/输出域与转换的表达能力分离，这为进一步分析提供了一个方便的框架。我们测量了 Imagenet 分类、​​COCO 对象检测、VOC 图像分割的性能。我们评估了准确率与乘加运算 (MAdd) 测量的运算次数以及参数数量之间的权衡
