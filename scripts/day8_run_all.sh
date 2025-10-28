#!/bin/bash

# Day 8 一键运行脚本
# 完成诊断、成本感知训练、超参数调优和综合对比

echo "======================================================================"
echo "Day 8 - PPO优化与诊断 (一键运行)"
echo "======================================================================"
echo ""

# 默认参数
EPISODES=${1:-10}
TIMESTEPS_COST_AWARE=${2:-100000}
TIMESTEPS_TUNED=${3:-150000}
MODE=${4:-"full"}  # full/quick

echo "运行模式: $MODE"
echo "评估轮数: $EPISODES"
echo "成本感知训练步数: $TIMESTEPS_COST_AWARE"
echo "调优训练步数: $TIMESTEPS_TUNED"
echo ""

# 检查Day 7模型是否存在
if [ ! -f "results/ppo_training/models/best_model/best_model.zip" ]; then
    echo "❌ 错误: Day 7的PPO模型不存在"
    echo "   请先完成Day 7的训练"
    exit 1
fi

echo "======================================================================" 
echo "步骤 1/4: 诊断分析"
echo "======================================================================"
echo ""

if [ "$MODE" = "quick" ]; then
    python3 scripts/day8_diagnose_ppo.py \
        --model results/ppo_training/models/best_model/best_model.zip \
        --episodes 3 \
        --quick
else
    python3 scripts/day8_diagnose_ppo.py \
        --model results/ppo_training/models/best_model/best_model.zip \
        --episodes $EPISODES
fi

if [ $? -ne 0 ]; then
    echo "❌ 诊断分析失败"
    exit 1
fi

echo ""
echo "✅ 诊断分析完成"
echo ""

# 暂停以查看诊断报告
echo "请查看诊断报告（按Enter继续）..."
read

echo "======================================================================"
echo "步骤 2/4: 成本感知PPO训练"
echo "======================================================================"
echo ""

python3 scripts/day8_train_cost_aware.py \
    --timesteps $TIMESTEPS_COST_AWARE \
    --cost-weight 2.0 \
    --quick-test

if [ $? -ne 0 ]; then
    echo "❌ 成本感知训练失败"
    exit 1
fi

echo ""
echo "✅ 成本感知训练完成"
echo ""

echo "======================================================================"
echo "步骤 3/4: 超参数调优PPO训练"
echo "======================================================================"
echo ""

if [ "$MODE" != "quick" ]; then
    python3 scripts/day8_train_tuned.py \
        --timesteps $TIMESTEPS_TUNED \
        --lr 0.0001 \
        --n-steps 4096 \
        --batch-size 128 \
        --quick-test
    
    if [ $? -ne 0 ]; then
        echo "⚠️  超参数调优训练失败，跳过此步骤"
    else
        echo ""
        echo "✅ 超参数调优训练完成"
        echo ""
    fi
else
    echo "⏭️  快速模式：跳过超参数调优"
    echo ""
fi

echo "======================================================================"
echo "步骤 4/4: 综合对比评估"
echo "======================================================================"
echo ""

if [ "$MODE" = "quick" ]; then
    python3 scripts/day8_compare_all.py --episodes 5
else
    python3 scripts/day8_compare_all.py --episodes $EPISODES
fi

if [ $? -ne 0 ]; then
    echo "❌ 综合对比失败"
    exit 1
fi

echo ""
echo "✅ 综合对比完成"
echo ""

echo "======================================================================"
echo "✅ Day 8 所有任务完成！"
echo "======================================================================"
echo ""
echo "📂 输出文件位置:"
echo "  - 诊断报告: results/ppo_diagnosis/"
echo "  - 成本感知模型: results/ppo_cost_aware/"
echo "  - 调优模型: results/ppo_tuned/"
echo "  - 综合对比: results/day8_comparison/"
echo ""
echo "💡 查看对比报告:"
echo "  cat results/day8_comparison/comparison_summary_*.txt"
echo ""
echo "🎯 根据对比结果，选择表现最好的模型用于后续分析"
echo ""