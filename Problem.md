找到了几个问题，从最严重到次要排列：

1. 模型在学"背景场直通"，而不是真正重建（最核心问题）

看归一化统计量：


bg_t_3d:    mean=20.879, std=8.935
label_t_3d: mean=20.889, std=8.932   ← 几乎一样！

bg_s_3d:    mean=34.196, std=0.942
label_s_3d: mean=34.194, std=0.944   ← 几乎一样！
Background 是 Glorys 往前推 7 天的数据，target 是当天 Glorys。7 天内海洋变化很小，所以两者统计上几乎相同。模型最省力的策略就是：直接把 bg_t_3d/bg_s_3d 原样输出，就能得到很低的 loss。这也解释了为什么 correlation=0.998 这么高——不是模型学得好，而是背景场本身就和 target 高度相关。SSS/SST/SLA 这三个表面观测实际上没有被有效利用。

2. SimpleModel 太浅，感受野太小


conv1: 43→64 (3x3)
conv2: 64→128 (3x3)
conv3: 128→64 (3x3)
conv4: 64→40 (3x3)
4 层 3x3 卷积，感受野只有 9x9 像素。对于 400x480 的海洋场，根本看不到大尺度结构（中尺度涡、锋面等），无法做有意义的空间重建。

3. Loss 函数被 mask 稀释


loss = criterion(outputs * masks, targets * masks)  # MSELoss默认mean
nn.MSELoss() 默认对所有元素取平均，包括陆地（mask=0）的像素。陆地像素两边都是 0，贡献 0 误差，但分母里算了它们，导致梯度信号被大量稀释。

4. 没有残差连接

模型应该学的是"在背景场基础上做修正"，但现在是从头学完整映射。加一个残差 output = bg + model(inputs) 会更合理，也更容易收敛。

总结：当前结果本质上是"背景场 + 微小扰动"，模型没有真正利用 SSS/SST/SLA 来改善重建。要改善效果，最关键的是：

换更深的模型（UNet 或带 skip connection 的架构），增大感受野
改 loss 为 masked mean：loss = ((outputs - targets) ** 2 * masks).sum() / masks.sum()
考虑加残差：让模型学习对背景场的修正量而不是直接预测绝对值