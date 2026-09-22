# 环境安装
## 设置独立缓存和虚拟环境路径
mkdir -p /data/Xixixi/VLA1/uv_envs
mkdir -p /data/Xixixi/VLA1/cache/uv
mkdir -p /data/Xixixi/VLA1/cache/huggingface
mkdir -p /data/Xixixi/VLA1/cache/uv_python

export UV_CACHE_DIR=/data/Xixixi/VLA1/cache/uv
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/Xixixi/VLA1/cache/huggingface
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=600
export UV_PYTHON_INSTALL_DIR=/data/Xixixi/VLA1/cache/uv_python

## 安装 OpenPI + 标准 LIBERO
cd /data/Xixixi/VLA1/RLinf

bash requirements/install.sh embodied \
  --platform nvidia \
  --model openpi \
  --env libero \
  --python 3.11.14 \
  --venv /data/Xixixi/VLA1/uv_envs/rlinf-openpi \
  --install-rlinf

## 安装 RLinf 维护的 LIBERO-Plus
UV_LINK_MODE=copy uv pip install "rlinf-liberoplus>=0.1.3"

## 下载标准 LIBERO 资产 & LIBERO-Plus 资产
libero-download-assets --skip-existing
export LIBERO_PLUS_ASSETS_REPO=RLinf/LIBERO-plus-assets
liberoplus-download-assets --skip-existing


# Baseline
## 修改
RLinf/evaluations/run_eval.sh: 修改日志默认文件夹
RLinf/rlinf/envs/sim/libero/libero_env.py: plus 评估可以指定类别

## pi0 libero
RLinf/evaluations/libero/libero_goal_pi0_eval.yaml

# feature/pi0-liberoplus-adapter-ppo
commit: 968d1ab04b006b7c4da8e4095f43fdc63cdaeaa9