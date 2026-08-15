# tensor-cp-decomposition

本仓库存放了 `CPALS.py` 和 `CPOPT.py` 两个 Python/PyTorch 代码文件，分别用于实现张量 CP 分解的 ALS（交替最小二乘）和 OPT（基于一阶梯度信息的优化）算法。

每个文件除存放 CP 分解的实现代码以外，还包含根据分解得到的因子矩阵复原张量、计算复原相对误差、真实数据可视化等功能的代码，可以处理合成张量数据（数值类型）和真实张量数据（图片、视频）。

---

This repository contains two Python/PyTorch code files, `CPALS.py` and `CPOPT.py`, which implement the ALS (Alternating Least Squares) and OPT (Optimization based on first-order gradient information) algorithms for Tensor CP Decomposition, respectively. 

In addition to the core CP decomposition algorithms, each file includes functionalities for reconstructing the tensor from the decomposed factor matrices, calculating the relative reconstruction error, and visualizing real data. The code is capable of processing both synthetic tensor data (numerical types) and real-world tensor data (images and videos).
