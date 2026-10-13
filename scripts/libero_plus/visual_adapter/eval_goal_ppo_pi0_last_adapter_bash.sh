# 1. Adapter only
EXPERIMENT_VARIANT=adapter_only \
EVAL_ALL=1 \
NUM_ENVS=25 \
RESET_RAY=1 \
bash /data/Xixixi/VLA1/RLinf/scripts/libero_plus/visual_adapter/eval_goal_ppo_pi0_last_adapter.sh


# 2. Adapter + Action Head
EXPERIMENT_VARIANT=adapter_action_head \
EVAL_ALL=1 \
NUM_ENVS=25 \
RESET_RAY=1 \
bash /data/Xixixi/VLA1/RLinf/scripts/libero_plus/visual_adapter/eval_goal_ppo_pi0_last_adapter.sh


# 3. Adapter + Action Expert
EXPERIMENT_VARIANT=adapter_action_expert \
EVAL_ALL=1 \
NUM_ENVS=25 \
RESET_RAY=1 \
ADAPTER_CKPT=/data/Xixixi/VLA1/RLinf/results/pi0/adapter_ppo/libero_goal_ppo_pi0_visual_adapter/adapter_action_expert/20260923-105224/pi0_last_adapter_action_expert_objects_layout_ppo_seed7/checkpoints/global_step_15/actor/trainable_model_state_dict/trainable_weights.pt \
bash /data/Xixixi/VLA1/RLinf/scripts/libero_plus/visual_adapter/eval_goal_ppo_pi0_last_adapter.sh

# 4. Adapter + Action Head + Action Expert
EXPERIMENT_VARIANT=adapter_action_head_action_expert \
EVAL_ALL=1 \
NUM_ENVS=25 \
RESET_RAY=1 \
bash /data/Xixixi/VLA1/RLinf/scripts/libero_plus/visual_adapter/eval_goal_ppo_pi0_last_adapter.sh
