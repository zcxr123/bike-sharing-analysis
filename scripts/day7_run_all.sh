#!/bin/bash
# Day 7 一键运行脚本
# 完成环境检查、PPO训练、评估对比全流程

set -e  # 遇到错误立即退出

echo "=========================================================================="
echo "Day 7: PPO训练与评估 - 一键运行脚本"
echo "=========================================================================="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目目录
PROJECT_DIR=~/bike-sharing-analysis
cd $PROJECT_DIR

# 参数设置（可修改）
TIMESTEPS=${1:-100000}      # 默认10万步
EVAL_EPISODES=${2:-10}      # 默认每场景10轮
MODE=${3:-full}             # full=完整, quick=快速, train=仅训练

echo -e "${YELLOW}配置:${NC}"
echo "  训练步数: $TIMESTEPS"
echo "  评估轮数: $EVAL_EPISODES"
echo "  运行模式: $MODE"
echo ""

# 步骤1: 环境检查
echo "=========================================================================="
echo "步骤1/3: 环境兼容性检查"
echo "=========================================================================="

if python3 scripts/day7_check_env.py; then
    echo -e "${GREEN}✅ 环境检查通过${NC}"
else
    echo -e "${RED}❌ 环境检查失败，请先解决问题${NC}"
    exit 1
fi

echo ""
read -p "按Enter键继续训练，或Ctrl+C退出..." 

# 步骤2: PPO训练
echo ""
echo "=========================================================================="
echo "步骤2/3: PPO训练"
echo "=========================================================================="

if [ "$MODE" != "eval" ]; then
    python3 scripts/day7_train_ppo.py \
        --timesteps $TIMESTEPS \
        --quick-test
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 训练完成${NC}"
    else
        echo -e "${RED}❌ 训练失败${NC}"
        exit 1
    fi
else
    echo "跳过训练（eval模式）"
fi

# 步骤3: 评估与对比
if [ "$MODE" != "train" ]; then
    echo ""
    echo "=========================================================================="
    echo "步骤3/3: PPO评估与对比"
    echo "=========================================================================="
    
    # 查找最新的best_model
    MODEL_PATH="results/ppo_training/models/best_model/best_model.zip"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}❌ 找不到训练好的模型: $MODEL_PATH${NC}"
        echo "可能训练时间太短，使用最终模型..."
        MODEL_PATH=$(ls -t results/ppo_training/models/ppo_final_*.zip 2>/dev/null | head -1)
        
        if [ -z "$MODEL_PATH" ]; then
            echo -e "${RED}❌ 未找到任何模型文件${NC}"
            exit 1
        fi
    fi
    
    echo "使用模型: $MODEL_PATH"
    
    # 根据模式选择评估参数
    if [ "$MODE" = "quick" ]; then
        # 快速模式：3个场景，5轮
        python3 scripts/day7_evaluate_ppo.py \
            --model $MODEL_PATH \
            --episodes 5 \
            --scenarios default sunny_weekday rainy_weekend
    else
        # 完整模式：5个场景，10轮
        python3 scripts/day7_evaluate_ppo.py \
            --model $MODEL_PATH \
            --episodes $EVAL_EPISODES
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 评估完成${NC}"
    else
        echo -e "${RED}❌ 评估失败${NC}"
        exit 1
    fi
    
    # 显示总结
    echo ""
    echo "=========================================================================="
    echo "评估总结"
    echo "=========================================================================="
    
    SUMMARY_FILE=$(ls -t results/ppo_evaluation/evaluation_summary_*.txt 2>/dev/null | head -1)
    if [ -f "$SUMMARY_FILE" ]; then
        cat $SUMMARY_FILE
    else
        echo "未找到总结文件"
    fi
fi

# 完成
echo ""
echo "=========================================================================="
echo -e "${GREEN}✅ Day 7 任务完成！${NC}"
echo "=========================================================================="

echo ""
echo "📁 输出文件:"
echo "  模型: results/ppo_training/models/"
echo "  日志: results/ppo_training/logs/"
echo "  评估: results/ppo_evaluation/"

echo ""
echo "📊 查看训练曲线:"
echo "  tensorboard --logdir results/ppo_training/logs"

echo ""
echo "📝 查看评估总结:"
echo "  cat results/ppo_evaluation/evaluation_summary_*.txt"

echo ""
echo "=========================================================================="