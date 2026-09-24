# 实验 1：只训练 Adapter
TRAIN_ACTION_HEAD=0 \
TRAIN_ACTION_EXPERT=0 \
EXPERIMENT_NAME="pi0_last_adapter_objects_layout_ppo_seed7" \
NUM_ENVS=8 \
ROLLOUT_EPOCH=2 \
GLOBAL_BATCH_SIZE=60 \
MICRO_BATCH_SIZE=1 \
POLICY_LR=5e-5 \
MAX_STEPS=200 \
SAVE_INTERVAL=20 \
VAL_INTERVAL=20 \
RESET_RAY=1 \
bash /data/Xixixi/VLA1/RLinf/scripts/libero_plus/visual_adapter/train_goal_ppo_pi0_last_adapter.sh


# 实验 2：Adapter + Action Head
TRAIN_ACTION_HEAD=1 \
TRAIN_ACTION_EXPERT=0 \
EXPERIMENT_NAME="pi0_last_adapter_action_head_objects_layout_ppo_seed7" \
NUM_ENVS=8 \
ROLLOUT_EPOCH=2 \
GLOBAL_BATCH_SIZE=60 \
MICRO_BATCH_SIZE=1 \
POLICY_LR=5e-5 \
MAX_STEPS=200 \
SAVE_INTERVAL=10 \
VAL_INTERVAL=10 \
RESET_RAY=1 \
bash /data/Xixixi/VLA1/RLinf/scripts/libero_plus/visual_adapter/train_goal_ppo_pi0_last_adapter.sh


# 实验 3：Adapter + Action Expert
TRAIN_ACTION_HEAD=0 \
TRAIN_ACTION_EXPERT=1 \
EXPERIMENT_NAME="pi0_last_adapter_action_expert_objects_layout_ppo_seed7" \
NUM_ENVS=8 \
ROLLOUT_EPOCH=2 \
GLOBAL_BATCH_SIZE=60 \
MICRO_BATCH_SIZE=1 \
POLICY_LR=1e-5 \
VALUE_LR=1e-4 \
MAX_STEPS=200 \
SAVE_INTERVAL=5 \
VAL_INTERVAL=5 \
RESET_RAY=1 \
bash /data/Xixixi/VLA1/RLinf/scripts/libero_plus/visual_adapter/train_goal_ppo_pi0_last_adapter.sh

# 实验 4：Adapter + Action Head + Action Expert
TRAIN_ACTION_HEAD=1 \
TRAIN_ACTION_EXPERT=1 \
EXPERIMENT_NAME="pi0_last_adapter_action_head_action_expert_objects_layout_ppo_seed7" \
NUM_ENVS=8 \
ROLLOUT_EPOCH=2 \
GLOBAL_BATCH_SIZE=60 \
MICRO_BATCH_SIZE=1 \
POLICY_LR=1e-5 \
MAX_STEPS=200 \
SAVE_INTERVAL=20 \
VAL_INTERVAL=20 \
RESET_RAY=1 \
bash /data/Xixixi/VLA1/RLinf/scripts/libero_plus/visual_adapter/train_goal_ppo_pi0_last_adapter.sh