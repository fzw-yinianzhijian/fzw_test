## Word2Vec 示例中使用的 PyTorch 组件说明

### 基础张量操作
- `torch.tensor(data, dtype)`：根据传入的 Python 数据构造张量，`dtype=torch.long` 常用于索引。
- `torch.sum(tensor, dim)`：在指定维度上求和，本例用于计算词向量点积。
- `torch.log(tensor)`：逐元素取对数，常配合概率或 Sigmoid 结果组成损失函数。
- `torch.mean(tensor)`：对张量求平均值，得到标量损失。
- `torch.sigmoid(tensor)`：逐元素执行 Sigmoid 激活，将实数映射到 `(0,1)` 概率区间。
- `torch.bmm(batch1, batch2)`：批量矩阵乘法，输入维度为 `(batch, n, m)` 与 `(batch, m, p)`。
- `tensor.unsqueeze(dim)`：在 `dim` 位置插入长度为 1 的维度，本例用于与负样本矩阵对齐。
- `tensor.transpose(dim0, dim1)`：交换两个维度的位置。
- `tensor.cpu()`：将张量从 GPU 拷贝到 CPU，便于与 NumPy 互操作。
- `tensor.numpy()`：将 CPU 张量转换为 NumPy 数组（需先在 CPU 上）。

### 神经网络模块
- `nn.Module`：所有模型的基类，提供参数注册、前向传播等基础设施。
- `nn.Embedding(num_embeddings, embedding_dim)`：查找表层，将词 ID 映射为可训练的稠密向量。
- `module.weight`：访问层的权重参数张量，`weight.data.uniform_(a, b)` 直接对参数赋初值。

### 优化与自动求导
- `torch.autograd.backward(loss)`：由 `loss.backward()` 触发自动求导，计算所有可训练参数的梯度。
- `optim.AdamW(parameters, lr)`：带权重衰减的 Adam 优化器，`zero_grad()` 清空梯度，`step()` 更新参数。

### CUDA 支持
- `torch.cuda.is_available()`：判断是否存在可用的 GPU。
- `torch.cuda.get_device_name(index)`：返回指定 GPU 的名称。

### 训练流程关键函数
- `model(center_tensor, context_tensor, neg_tensor)`：调用 `forward`，返回正负样本分数。
- `loss_fn(positive_scores, negative_scores)`：自定义损失函数，组合正样本与负样本的对数似然。
- `optimizer.zero_grad()`：梯度置零，准备新的反向传播。
- `loss.backward()`：自动求导，计算梯度。
- `optimizer.step()`：依据梯度更新模型参数。

