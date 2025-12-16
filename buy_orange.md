# 理解链式法则的反向传播机制

| 复合函数的求导只能用链式法则

## 举例

函数$y=f(g(x))$,若要求$dy/dx$,则必须根据公式$dy/dx=dy/dg*dg/dx$
dy/dg是“后面的”,dg/dx是“前面的”

## 买橘子的传播

苹果单价：100 元，买 2 个
橘子单价：50 元，买 3 个
折扣：0.9折
求总价，即*“对每个变量的梯度”*

| 变量 | 含义   | 数值  |
| -- | ---- | --- |
| a  | 苹果单价 | 100 |
| b  | 苹果数量 | 2   |
| c  | 橘子单价 | 50 |
| d  | 橘子数量 | 3   |
| t  |折扣   | 0.9 |
计算式为：
```text
a × b = apple_sum = 200
c × d = orange_sum = 450
apple_sum + orange_sum = subtotal = 650
subtotal × t = total = 715
```

| 变量      | 梯度  | 含义解释               |
| ------- | --- | ------------------ |
| a（苹果单价） | 1.8 | 单价涨 1 元，总价涨 1.8 元  |
| b（苹果数量） | 90 | 多买 1 个苹果，总价涨 90 元 |
| c（橘子单价） | 2.7 | 单价涨 1 元，总价涨 2.7 元  |
| d（橘子数量） | 45 | 多买 1 个橘子，总价涨 45 元 |
| t（折扣）   | 350 | 折扣每增加0.1，总价涨 35    |


非常好，这一步**正是“从会用 → 真正理解反向传播”的分水岭**。
下面我会**不跳步、不过度抽象**，把**两层神经网络（含 ReLU）**的梯度公式**从定义一步一步推导出来**。

> 我会刻意使用你已经熟悉的符号和“房价预测”的设定。

---

# 一、模型与符号约定（先固定，不然后面会乱）

### 1?? 数据与维度

* 样本数：(N)
* 输入维度：(D=13)
* 隐藏层维度：(H)
* 输出维度：1

$$
X \in \mathbb{R}^{N \times D},\quad
y \in \mathbb{R}^{N \times 1}
$$

---

### 2?? 两层网络定义（含 ReLU）

$$
\begin{aligned}
Z_1 &= X W_1 + b_1 \quad &(N\times H)\
A_1 &= \text{ReLU}(Z_1) \quad &(N\times H)\
Z_2 &= A_1 W_2 + b_2 \quad &(N\times 1)\
\hat y &= Z_2
\end{aligned}
$$

参数：

$$
W_1 \in \mathbb{R}^{D\times H},;
b_1 \in \mathbb{R}^{1\times H},;
W_2 \in \mathbb{R}^{H\times 1},;
b_2 \in \mathbb{R}^{1\times 1}
$$

---

### 3?? 损失函数（MSE）

$$
L = \frac{1}{2N}\sum_{i=1}^{N}(\hat y_i - y_i)^2
$$

（加 (1/2) 是为了后面好看，不影响本质）

---

# 二、反向传播的起点：(\partial L / \partial \hat y)

对单个样本：

$$
\ell_i = \frac12 (\hat y_i - y_i)^2
$$

求导：

$$
\frac{\partial \ell_i}{\partial \hat y_i}
= \hat y_i - y_i
$$

对所有样本：

$$
\boxed{
\frac{\partial L}{\partial Z_2}
= \frac{1}{N}(\hat y - y)
}
$$

记作：

$$
\delta_2 \equiv \frac{\partial L}{\partial Z_2}
\quad (N\times1)
$$

---

# 三、第二层参数的梯度推导（W?、b?）

---

## 1?? 推导 (\partial L / \partial W_2)

### 展开 Z?：

$$
Z_2 = A_1 W_2 + b_2
$$

对矩阵中第 (j) 个权重 (W_{2,j})：

$$
\frac{\partial Z_{2,i}}{\partial W_{2,j}}
= A_{1,i j}
$$

使用链式法则：

$$
\frac{\partial L}{\partial W_{2,j}}
===================================

\sum_{i=1}^{N}
\frac{\partial L}{\partial Z_{2,i}}
\frac{\partial Z_{2,i}}{\partial W_{2,j}}
=========================================

\sum_i \delta_{2,i} A_{1,i j}
$$

写成矩阵形式：

$$
\boxed{
\frac{\partial L}{\partial W_2}
= A_1^T \delta_2
}
\quad (H\times1)
$$

---

## 2?? 推导 (\partial L / \partial b_2)

$$
\frac{\partial Z_{2,i}}{\partial b_2} = 1
$$

$$
\boxed{
\frac{\partial L}{\partial b_2}
= \sum_i \delta_{2,i}
}
$$

---

# 四、误差反传到隐藏层（核心）

我们现在要算：

$$
\frac{\partial L}{\partial Z_1}
$$

---

## 1?? 先算 (\partial L / \partial A_1)

$$
Z_2 = A_1 W_2 + b_2
$$

$$
\boxed{
\frac{\partial L}{\partial A_1}
= \delta_2 W_2^T
}
\quad (N\times H)
$$

---

## 2?? 再过 ReLU（关键非线性）

$$
A_1 = \text{ReLU}(Z_1)
$$

ReLU 导数：

$$
\text{ReLU}'(Z_1) =
\begin{cases}
1 & Z_1 > 0\
0 & Z_1 \le 0
\end{cases}
$$

逐元素相乘：

$$
\boxed{
\delta_1
\equiv
\frac{\partial L}{\partial Z_1}
===============================

(\delta_2 W_2^T)
\odot
\mathbf{1}_{Z_1>0}
}
$$

---

# 五、第一层参数梯度推导（W?、b?）

---

## 1?? 推导 (\partial L / \partial W_1)

$$
Z_1 = X W_1 + b_1
$$

对 (W_{1,kl})：

$$
\frac{\partial Z_{1,i l}}{\partial W_{1,kl}}
= X_{ik}
$$

链式法则：

$$
\frac{\partial L}{\partial W_{1,kl}}
====================================

\sum_i
\frac{\partial L}{\partial Z_{1,i l}}
\frac{\partial Z_{1,i l}}{\partial W_{1,kl}}
============================================

\sum_i \delta_{1,i l} X_{ik}
$$

矩阵形式：

$$
\boxed{
\frac{\partial L}{\partial W_1}
= X^T \delta_1
}
\quad (D\times H)
$$

---

## 2?? 推导 (\partial L / \partial b_1)

$$
\frac{\partial Z_{1,i l}}{\partial b_{1,l}} = 1
$$

$$
\boxed{
\frac{\partial L}{\partial b_1}
= \sum_i \delta_{1,i}
}
$$

---

# 六、最终梯度公式总览（一页记住）

### 输出层：

$$
\delta_2 = \frac{1}{N}(\hat y - y)
$$

$$
\nabla W_2 = A_1^T \delta_2
$$

$$
\nabla b_2 = \sum \delta_2
$$

---

### 隐藏层：

$$
\delta_1 = (\delta_2 W_2^T)\odot \mathbf{1}_{Z_1>0}
$$

$$
\nabla W_1 = X^T \delta_1
$$

$$
\nabla b_1 = \sum \delta_1
$$

---

# 七、为什么这些公式“长这样”？一句话理解版

* **误差 (\delta)**：从 loss 向前一层“追责”
* **转置 (X^T, A_1^T)**：
  把“样本维”累加掉
* **ReLU 掩码**：
  决定哪些神经元“参与学习”

---

# 八、你已经到什么水平了？

如果你能**看懂并复述**这套推导，你已经：

* ? 真正理解反向传播
* ? 能手写任意 MLP
* ? 不再依赖“背公式”
