# Reference results

`qwen2.5-1.5b/` 保存当前 MXINT4-B32 full-A/full-G 六格实验的小型汇总表和图。

`legacy_diagonal/` 保存此前 INT4 对角统计实验的 PPL 汇总与 rank 对比。

这些文件只用于：

- 检查新代码输出列、数量级和曲线方向；
- 记录本地实验基线；
- 比较同一模型、同一 tokenizer、同一数据协议下的趋势。

它们不能作为 Llama 3、其他 Qwen 尺寸或不同 tokenizer 的绝对 PPL 基准。大体积逐窗口结果、A/G 矩阵、量化权重和校正因子没有放入仓库。
