# 基于GAT的社交网络关键节点预测与传播模拟

本项目使用图注意力网络(GAT)对社交网络中的关键节点进行预测，并通过传播模型模拟验证预测效果。

## 项目结构

```
project/
├── data/                      # 数据目录
├── src/                       # 源代码
├── notebooks/                 # Jupyter notebooks
├── tests/                     # 单元测试
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明文档
```

## 环境配置

1. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 数据准备：
   - 将数据集放置在 `data/raw` 目录下

2. 模型训练：
```bash
python src/train.py
```

3. 运行测试：
```bash
pytest tests/
```

## 主要功能

- 基于GAT的关键节点预测
- 社交网络传播模拟
- 与传统方法的对比分析
- 可视化分析工具

## 依赖说明

主要依赖包括：
- PyTorch：深度学习框架
- PyTorch Geometric：图神经网络库
- NetworkX：图分析工具
- 其他数据处理和可视化工具

详细依赖列表请参考 `requirements.txt` 