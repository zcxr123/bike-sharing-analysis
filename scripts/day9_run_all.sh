#!/bin/bash

# Day 9 一键运行脚本
# 完成决策分析、可视化生成和报告编写

echo "======================================================================"
echo "Day 9 - 深化分析与成果展示 (一键运行)"
echo "======================================================================"
echo ""

# 默认参数
MODE=${1:-"full"}  # full/quick
EPISODES=${2:-10}

echo "运行模式: $MODE"
echo "分析轮数: $EPISODES"
echo ""

# 检查Day 8数据是否存在
if [ ! -f "results/ppo_cost_aware/models/best_model/best_model.zip" ]; then
    echo "❌ 错误: Day 8的成本感知模型不存在"
    echo "   请先完成Day 8的训练"
    exit 1
fi

if [ ! -d "results/day8_comparison" ]; then
    echo "❌ 错误: Day 8的对比数据不存在"
    echo "   请先完成Day 8的评估"
    exit 1
fi

echo "======================================================================"
echo "步骤 1/3: 决策可解释性分析"
echo "======================================================================"
echo ""

if [ "$MODE" = "quick" ]; then
    python3 scripts/day9_analyze_decisions.py \
        --model results/ppo_cost_aware/models/best_model/best_model.zip \
        --episodes 3 \
        --quick
else
    python3 scripts/day9_analyze_decisions.py \
        --model results/ppo_cost_aware/models/best_model/best_model.zip \
        --episodes $EPISODES
fi

if [ $? -ne 0 ]; then
    echo "❌ 决策分析失败"
    exit 1
fi

echo ""
echo "✅ 决策分析完成"
echo ""

echo "======================================================================"
echo "步骤 2/3: 生成可视化图表"
echo "======================================================================"
echo ""

if [ "$MODE" = "quick" ]; then
    python3 scripts/day9_generate_plots.py --essential-only
else
    python3 scripts/day9_generate_plots.py
fi

if [ $? -ne 0 ]; then
    echo "❌ 可视化生成失败"
    exit 1
fi

echo ""
echo "✅ 可视化生成完成"
echo ""

echo "======================================================================"
echo "步骤 3/3: 生成报告"
echo "======================================================================"
echo ""

if [ "$MODE" = "quick" ]; then
    python3 scripts/day9_generate_reports.py --executive-summary
else
    python3 scripts/day9_generate_reports.py
fi

if [ $? -ne 0 ]; then
    echo "❌ 报告生成失败"
    exit 1
fi

echo ""
echo "✅ 报告生成完成"
echo ""

echo "======================================================================"
echo "✅ Day 9 所有任务完成！"
echo "======================================================================"
echo ""
echo "📂 输出文件位置:"
echo "  - 决策分析: results/day9_analysis/"
echo "  - 可视化图表: results/day9_visualizations/"
echo "  - 业务报告: results/day9_reports/"
echo ""
echo "💡 查看关键文件:"
echo "  # 决策分析报告"
echo "  cat results/day9_analysis/decision_analysis_report_*.txt"
echo ""
echo "  # 可视化图表"
echo "  ls -lh results/day9_visualizations/*.png"
echo ""
echo "  # 业务报告"
echo "  cat results/day9_reports/business_report_*.md"
echo ""
echo "🎯 Day 9成功展示了Day 8的优秀成果！"
echo ""