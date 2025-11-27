# FCA_GNN
FCA+GNN处理分类任务

### 1. 环境与数据准备 (基础阶段)

这部分主要用于确保代码能跑通，并打通数据加载流程。

- **`1_Environment_Check.ipynb`**
  - **作用**：**环境自检**。检查 PyTorch, CUDA, PyG 版本，并运行一个极简的 GNN demo 确保库安装正确。
- **`2_Data_Loading_and_Graph_Creation.ipynb`**
  - **作用**：**数据加载演示**。演示如何读取清洗后的数据 (`.cleaned.csv`) 和 FCA 生成的邻接矩阵 (`_A_plus.csv`)，并将它们封装成 PyG 的 `Data` 对象。
- **`context.ipynb`**
  - **作用**：**格式修正**。简单的脚本，用于去除数据行末尾多余的逗号，生成 `.cleaned.csv`。
- **`data_process.ipynb`**
  - **作用**：**通用预处理工具**。包含读取数据、删除列、缺失值处理、标签编码 (LabelEncoder) 等通用函数的工具箱。



### 2. 基线模型构建与迭代 (单分支 GNN)



这部分是在 Iris 数据集上逐步完善单分支 FCA-GNN 模型的训练流程。

- **`3_Baseline_FCA-GNN_Training_on_Iris.ipynb`**
  - **作用**：**基线训练**。首次在 Iris 数据集上跑通完整的 GCN 训练流程，使用阈值化后的正概念格邻接矩阵。
- **`4_Ablation_Continuous_Features_Experiment.ipynb`**
  - **作用**：**特征消融实验**。将节点特征从“FCA二元属性”替换为“原始连续数值特征”，对比不同特征输入对效果的影响。
- **`5_Training_with_TensorBoard_Logging.ipynb`**
  - **作用**：**工程优化**。在训练循环中加入 TensorBoard 日志记录 (`SummaryWriter`)，用于可视化 Loss 和 Accuracy 曲线。
- **`6_Adding_Validation_Set_and_Metrics.ipynb`**
  - **作用**：**评估完善**。将数据集划分从“训练/测试”改为更标准的“训练/验证/测试” (60%/20%/20%)，防止过拟合测试集。



### 3. 单分支模型变体研究 (Series 7)



这部分深入探索了**不同类型的概念格结构**作为图结构的效果。

- **`7_Final_FCA-GNN_Single_Branch_Model.ipynb`**
  - **作用**：**单分支最终版**。整合了前面所有优化的标准单分支 GCN 模型 (使用 A_plus)。
- **`7_2_UG.ipynb`**
  - **作用**：**无向图变体**。使用 `A_plus_UG` (Undirected Graph) 邻接矩阵进行实验。
- **`7_3_A_negative_UG.ipynb`**
  - **作用**：**负概念格(无向)**。探索使用**负概念格** (Negative Concept Lattice，基于对象**不具有**某属性构建) 的无向图结构。
- **`7_4_A_negative_DG.ipynb`**
  - **作用**：**负概念格(有向)**。探索使用**负概念格**的有向图结构 (Directed Graph)。



### 4. 对照实验 (Series 8)



- **`8_Control_Experiment_with_Random_Graph.ipynb`**
  - **作用**：**随机图对照**。生成与概念格边数相同的**随机图**来训练 GNN。
  - **目的**：证明模型效果好是因为概念格捕捉了真实结构，而不是仅仅因为有了图结构。



### 5. 双分支融合模型 (Series 9)



这部分尝试结合两种不同的图结构来提升性能。

- **`9_Dual_Branch_CLGCN_Full_Model.ipynb`**
  - **作用**：**FCA + 余弦相似度**。分支一用 FCA 概念格图，分支二用基于原始特征计算的**余弦相似度图**，融合两者特征。
- **`9_2_branch3C.ipynb`**
  - **作用**：**正概念格 + 负概念格**。双分支模型，同时利用对象“具有属性” (Positive) 和“不具有属性” (Negative) 的结构信息。
- **`9_2_experiment.ipynb`**
  - **作用**：**Zoo 数据集实验**。将 `9_2` 的架构应用在 `Zoo` 数据集上，进行了适配。
- **`9_2_experiment_hparas.ipynb`**
  - **作用**：**超参数记录**。在 `9_2` 基础上增强了实验记录，将超参数 (阈值、LR等) 写入 TensorBoard 以便对比调优。



### 6. 高级模型探索 (Transformer)



尝试引入更先进的图神经网络算子。

- **`10_Dual_Branch_Graph_Transformer.ipynb`**
  - **作用**：**Graph Transformer**。将 `GCNConv` 替换为 `TransformerConv`，引入多头注意力机制 (Multi-head Attention) 处理正负概念格双分支。
- **`11_Transformer_with_Positional_Encoding.ipynb`**
  - **作用**：**位置编码增强**。在 Transformer 基础上，计算拉普拉斯特征向量 (Laplacian Eigenvectors) 作为**位置编码 (Positional Encoding)** 并拼接到节点特征中，增强模型对图结构的感知能力。