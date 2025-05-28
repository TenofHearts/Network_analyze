import os
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据相关路径
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 输出相关路径
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

# 创建必要的目录
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据集URL
DATASET_URLS = {
    "facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "twitter": "https://snap.stanford.edu/data/twitter_combined.txt.gz",
    "epinions": "https://snap.stanford.edu/data/soc-Epinions1.txt.gz",
}
