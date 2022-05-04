# FederatedLearningOnRaspberry
可部署在硬件平台上进行多机训练的同步联邦学习框架

# 项目介绍
- 联邦平均算法（Fedavg）
- MNIST数据集（独立同分布）
- 逻辑回归模型与卷积神经网络模型（Logistic&CNN）
- 兼顾文件传输功能（便于收集客户端上的数据）
- 代码包含原有冗余部分未删除，部分代码块未使用
- 测试集数据集请使用MNIST_test2.pkl，可兼容两种神经网络模型
- 环境配置请参考environment.txt

# 使用须知
- 客户端与服务器程序为client_demo_filetrans.py,server_demo_filetrans.py
- 请先运行服务器端文件再运行客户端文件
- 默认客户端数量为20，请在服务器端文件中修改n_nodes的值
- 联邦学习框架各项设置在服务器端文件read_options中修改

如有任何问题，请在issues区中提出
