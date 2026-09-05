# Method

## 1. Weighted low-rank correction

对线性层权重 `W ∈ R^(d_out×d_in)`，MXINT4 权重为 `Wq`，量化误差定义为

```math
E = W - W_q.
```

用 rank-`r` 校正 `ΔW = LR^T`，其中 `L ∈ R^(d_out×r)`、`R ∈ R^(d_in×r)`。实验求解

```math
\min_{\operatorname{rank}(\Delta W)\le r}
\left\|G^{1/2}(E-\Delta W)A^{1/2}\right\|_F^2.
```

`A` 是该线性层输入激活的 token-wise 二阶矩，`G` 是该线性层输出梯度的 token-wise 二阶矩：

```math
A=\frac{1}{N}\sum_{t=1}^N x_t x_t^T,
\qquad
G=\frac{1}{N}\sum_{t=1}^N g_t g_t^T,
```

其中 `g_t = ∂ℓ_t/∂y_t`，`ℓ_t` 为 clean teacher 上的真实 next-token 交叉熵，不使用 pseudo label、梯度裁剪或梯度放大。

## 2. `G^(1/2)` 的构造

先对采集矩阵对称化并加 trace-scaled damping：

```math
\bar G=\frac{G+G^T}{2},
\qquad
\delta_G=\lambda_G\frac{\operatorname{tr}(\bar G)}{d_{out}},
\qquad
G_\lambda=\bar G+\delta_G I.
```

### Eigh 路径

若

```math
\bar G=U_G\operatorname{diag}(\gamma)U_G^T,
```

先把数值误差导致的微小负特征值截断到 0，再构造

```math
G_\lambda^{1/2}
=U_G\operatorname{diag}\!\left(\sqrt{\max(\gamma,0)+\delta_G}\right)U_G^T,
```

```math
G_\lambda^{-1/2}
=U_G\operatorname{diag}\!\left((\max(\gamma,0)+\delta_G)^{-1/2}\right)U_G^T.
```

`A^(1/2)` 同理。该路径与当前 Qwen2.5-1.5B 正式实验一致。

### Cholesky 路径

对更宽的 Llama/Qwen MLP，可设 `statistics.root_method: cholesky`。若

```math
G_\lambda=C_GC_G^T,
\qquad
A_\lambda=C_AC_A^T,
```

则等价目标为

```math
\left\|C_G^T(E-\Delta W)C_A\right\|_F^2.
```

Cholesky 与对称平方根只是合法因子分解不同，优化的加权二次型相同。为了严格复核旧实验，优先使用 `eigh`；为了大型模型可行性，可在独立运行中使用 `cholesky`，并记录 resolved config。

## 3. Closed-form rank-r solution

定义 whitened quantization error：

```math
M=G^{1/2}EA^{1/2}.
```

对 `M` 做截断 SVD：

```math
M=U\Sigma V^T,
\qquad
M_r=U_r\Sigma_rV_r^T.
```

对称平方根路径下，可取

```math
L=G^{-1/2}U_r\Sigma_r^{1/2},
\qquad
R=A^{-1/2}V_r\Sigma_r^{1/2},
```

因此

```math
\Delta W_r=LR^T=G^{-1/2}M_rA^{-1/2}.
```

各 rank 直接使用最大 rank 因子的列前缀，避免每个 rank 重新求解导致排序不一致。

## 4. Full-G 与 diagonal-G

`G_D = diag(G)` 只保留每个输出通道的梯度能量，忽略不同输出通道之间的协方差。其平方根和逆平方根都是逐元素缩放：

```math
G_D^{1/2}=\operatorname{diag}(\sqrt{g_{ii}+\delta_G}),
\qquad
G_D^{-1/2}=\operatorname{diag}((g_{ii}+\delta_G)^{-1/2}).
```

`G_F = G` 保留全部非对角相关性，白化会旋转输出空间，不只是改变每个通道的尺度。于是：

- diagonal-G 的代价和存储约为 `O(d_out)`；
- full-G 的矩阵存储为 `O(d_out^2)`，分解为 `O(d_out^3)`；
- full-G 只有在额外相关性带来的方向选择改善能传递到端到端 PPL 时才值得。

同样，`A_D` 与 `A_F` 分别忽略或保留输入通道的协方差。六格设计把 A 的选择、G 的选择及两者交互分开。

## 5. Evaluation

部署计算为

```math
y=W_qx+L(R^Tx)+b.
```

PPL 通过全局 NLL 聚合：

```math
\operatorname{PPL}=\exp\left(\frac{\sum_w \operatorname{NLL}_w}{\sum_w N_w}\right).
```

不同模型必须重新 tokenize 并冻结 calibration、WikiText-2 与 C4 窗口；不能把一个 tokenizer 的 token id 缓存用于另一个模型。
