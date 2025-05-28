import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path

from models.gat import GATModel
from models.propagation import IndependentCascade
from utils.metrics import calculate_spread_metrics, calculate_node_importance_scores
from utils.visualization import plot_propagation_history, plot_node_importance
from data.dataset import SNAPDataset


def save_model(model, optimizer, epoch, loss, save_path):
    """
    保存模型和训练状态

    参数:
        model: GAT模型
        optimizer: 优化器
        epoch: 当前训练轮数
        loss: 当前损失值
        save_path: 保存路径
    """
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        save_path,
    )
    print(f"模型已保存到: {save_path}")


def load_model(model, optimizer, load_path, device="cpu"):
    """
    加载模型和训练状态

    参数:
        model: GAT模型
        optimizer: 优化器
        load_path: 加载路径
        device: 设备

    返回:
        epoch: 训练轮数
        loss: 损失值
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"模型已从 {load_path} 加载")
    return epoch, loss


def train_model(
    model,
    data,
    optimizer,
    criterion,
    epochs=100,
    device="cpu",
    save_path=None,
    load_path=None,
):
    """
    训练GAT模型

    参数:
        model: GAT模型
        data: 图数据
        optimizer: 优化器
        criterion: 损失函数
        epochs: 训练轮数
        device: 训练设备
        save_path: 模型保存路径
        load_path: 模型加载路径
    """
    model = model.to(device)
    data = data.to(device)
    model.train()

    # 记录训练过程
    train_losses = []
    start_epoch = 0

    # 如果提供了加载路径，加载模型
    if load_path and os.path.exists(load_path):
        start_epoch, last_loss = load_model(model, optimizer, load_path, device)
        train_losses.append(last_loss)
        print(f"从第 {start_epoch} 轮继续训练")

    for epoch in tqdm(range(start_epoch, epochs)):
        optimizer.zero_grad()

        # 前向传播
        out = model(data.x, data.edge_index)

        # 计算损失
        loss = criterion(out, data.y)

        # 反向传播
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

            # 每10轮保存一次模型
            if save_path:
                save_model(model, optimizer, epoch + 1, loss.item(), save_path)

    return train_losses


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练GAT模型进行关键节点预测")
    parser.add_argument(
        "--dataset",
        type=str,
        default="facebook",
        choices=["facebook", "twitter", "epinions"],
        help="选择数据集",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--hidden_channels", type=int, default=32, help="隐藏层维度")
    parser.add_argument("--heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--top_k", type=int, default=10, help="选择的关键节点数量")
    parser.add_argument("--n_simulations", type=int, default=100, help="IC模型模拟次数")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="模型保存路径",
    )
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="模型加载路径",
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)

    # 设置默认的模型保存路径
    if args.save_path is None:
        args.save_path = f"outputs/models/{args.dataset}_gat_model.pt"

    # 加载数据集
    dataset = SNAPDataset(args.dataset, n_simulations=args.n_simulations)
    data = dataset.to_pyg_data()

    # 模型参数
    in_channels = data.x.size(1)  # 特征维度
    hidden_channels = args.hidden_channels
    out_channels = 1  # 输出维度（节点重要性分数）

    # 创建模型
    model = GATModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=args.heads,
    )

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 损失函数
    criterion = nn.MSELoss()

    print(f"数据集: {args.dataset}")
    print(f"节点数: {data.x.size(0)}")
    print(f"边数: {data.edge_index.size(1)}")
    print(f"特征维度: {in_channels}")
    print(f"IC模拟次数: {args.n_simulations}")
    print(f"训练设备: {args.device}")
    print(f"模型保存路径: {args.save_path}")
    if args.load_path:
        print(f"模型加载路径: {args.load_path}")
    print("\n模型结构：")
    print(model)

    # 训练模型
    print("\n开始训练...")
    train_losses = train_model(
        model,
        data,
        optimizer,
        criterion,
        epochs=args.epochs,
        device=args.device,
        save_path=args.save_path,
        load_path=args.load_path,
    )

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title("training loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"outputs/{args.dataset}_training_loss.png")

    # 评估模型
    model.eval()
    with torch.no_grad():
        data = data.to(args.device)
        predictions = model(data.x, data.edge_index)
        predictions = predictions.cpu().numpy()
        importance_scores = calculate_node_importance_scores(predictions)

        # 获取top-k节点
        top_k_nodes = np.argsort(importance_scores)[-args.top_k :]
        print("\nTop-K关键节点：")
        for i, node_id in enumerate(top_k_nodes):
            print(
                f"{i+1}. 节点 {node_id} (重要性分数: {importance_scores[node_id]:.4f})"
            )

        # 可视化节点重要性
        graph = dataset.load_graph()
        plot_node_importance(
            graph,
            importance_scores,
            title=f"{args.dataset} node importance distribution",
        )
        plt.savefig(f"outputs/{args.dataset}_importance.png")
        plt.close()  # 关闭图形，避免内存泄漏

        # 传播模拟
        propagation_model = IndependentCascade(graph)

        # 进行多次传播模拟
        propagation_results = []
        for _ in range(10):
            result = propagation_model.simulate(top_k_nodes.tolist())
            propagation_results.append(result)

        # 计算传播指标
        metrics = calculate_spread_metrics(propagation_results)
        print("\n传播效果指标：")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # 可视化传播过程
        plot_propagation_history(
            propagation_results, title=f"{args.dataset} propagation process"
        )
        plt.savefig(f"outputs/{args.dataset}_propagation.png")
        plt.close()  # 关闭图形，避免内存泄漏


if __name__ == "__main__":
    main()
