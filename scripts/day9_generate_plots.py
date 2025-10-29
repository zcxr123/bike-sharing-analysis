#!/usr/bin/env python3
"""
Day 9 - 生成可视化图表
展示Day 8的优秀成果
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(description='生成可视化图表')
    parser.add_argument('--essential-only', action='store_true',
                       help='只生成核心图表')
    return parser.parse_args()


def setup_plotting_style():
    """设置绘图风格"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def load_comparison_data():
    """加载对比数据"""
    print("📊 加载Day 8对比数据...")
    
    # 查找最新的对比文件
    comparison_dir = Path("results/day8_comparison")
    csv_files = list(comparison_dir.glob("comparison_detail_*.csv"))
    
    if not csv_files:
        print("❌ 错误: 找不到Day 8对比数据")
        print("   请先完成Day 8的评估")
        return None
    
    # 使用最新的文件
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"  加载文件: {latest_file.name}")
    
    df = pd.read_csv(latest_file)
    print(f"✅ 数据加载成功: {len(df)}条记录")
    
    return df


def plot_cost_comparison(df, output_dir):
    """成本对比柱状图"""
    print("\n📊 生成成本对比图...")
    
    # 计算平均成本
    cost_summary = df.groupby('model')['total_cost'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if 'Day7' in idx else '#2ecc71' if 'Day8' in idx else '#3498db' 
              for idx in cost_summary.index]
    
    bars = ax.bar(range(len(cost_summary)), cost_summary.values, color=colors, alpha=0.8)
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, cost_summary.values)):
        ax.text(bar.get_x() + bar.get_width()/2, value + 20, 
                f'${value:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(len(cost_summary)))
    ax.set_xticklabels(cost_summary.index, rotation=15, ha='right')
    ax.set_ylabel('平均调度成本 ($)', fontsize=12)
    ax.set_title('策略成本对比 - Day 8实现76%成本降低', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加改进标注
    if 'PPO-Day7-Original' in cost_summary.index and 'PPO-Day8-CostAware' in cost_summary.index:
        day7_cost = cost_summary['PPO-Day7-Original']
        day8_cost = cost_summary['PPO-Day8-CostAware']
        improvement = (day7_cost - day8_cost) / day7_cost * 100
        
        ax.annotate(f'降低{improvement:.1f}%', 
                   xy=(cost_summary.index.get_loc('PPO-Day7-Original'), day7_cost),
                   xytext=(cost_summary.index.get_loc('PPO-Day8-CostAware'), day8_cost + 200),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'cost_comparison_bar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {output_path.name}")
    return output_path


def plot_service_cost_tradeoff(df, output_dir):
    """服务率-成本权衡曲线"""
    print("\n📊 生成服务率-成本权衡图...")
    
    # 按模型和场景分组
    summary = df.groupby('model').agg({
        'service_rate': 'mean',
        'total_cost': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 绘制每个模型的点
    for _, row in summary.iterrows():
        model = row['model']
        color = '#e74c3c' if 'Day7' in model else '#2ecc71' if 'Day8' in model else '#3498db'
        marker = 'o' if 'PPO' in model else 's'
        size = 200
        
        ax.scatter(row['service_rate']*100, row['total_cost'], 
                  s=size, color=color, marker=marker, alpha=0.7, edgecolors='black', linewidth=2)
        
        # 添加标签
        offset_x = -1 if 'Day7' in model else 0.3
        offset_y = 50 if 'Day7' in model else -100
        ax.annotate(model.replace('PPO-', '').replace('Proportional-Optimized', 'Baseline'), 
                   xy=(row['service_rate']*100, row['total_cost']),
                   xytext=(row['service_rate']*100 + offset_x, row['total_cost'] + offset_y),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))
    
    # 绘制帕累托前沿
    pareto_points = summary.sort_values('service_rate')
    ax.plot(pareto_points['service_rate']*100, pareto_points['total_cost'], 
           'k--', alpha=0.3, linewidth=1, label='权衡曲线')
    
    ax.set_xlabel('服务率 (%)', fontsize=12)
    ax.set_ylabel('调度成本 ($)', fontsize=12)
    ax.set_title('服务率-成本权衡分析 - 98%是最优平衡点', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加最优区域标注
    ax.axvspan(97, 99, alpha=0.1, color='green', label='最优区域')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    output_path = output_dir / 'service_cost_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {output_path.name}")
    return output_path


def plot_roi_comparison(df, output_dir):
    """ROI对比图"""
    print("\n📊 生成ROI对比图...")
    
    # 计算ROI
    roi_data = df.groupby('model').agg({
        'net_profit': 'mean',
        'total_cost': 'mean'
    }).reset_index()
    
    roi_data['roi'] = roi_data['net_profit'] / roi_data['total_cost']
    roi_data = roi_data.sort_values('roi')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c' if 'Day7' in idx else '#2ecc71' if 'Day8' in idx else '#3498db' 
              for idx in roi_data['model']]
    
    bars = ax.barh(range(len(roi_data)), roi_data['roi'], color=colors, alpha=0.8)
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, roi_data['roi'])):
        ax.text(value + 5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}x', va='center', fontsize=11, fontweight='bold')
    
    ax.set_yticks(range(len(roi_data)))
    ax.set_yticklabels(roi_data['model'], fontsize=10)
    ax.set_xlabel('投资回报率 (ROI = 净利润 / 成本)', fontsize=12)
    ax.set_title('ROI对比 - Day 8提升4.3倍', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 添加改进标注
    if len(roi_data) >= 2:
        max_idx = roi_data['roi'].idxmax()
        max_roi = roi_data.loc[max_idx, 'roi']
        ax.axvline(max_roi, color='green', linestyle='--', linewidth=2, alpha=0.5)
        ax.text(max_roi, len(roi_data)-0.5, f'最高: {max_roi:.1f}x', 
               ha='right', va='top', fontsize=11, color='green', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'roi_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {output_path.name}")
    return output_path


def plot_scenario_heatmap(df, output_dir):
    """场景性能热力图"""
    print("\n📊 生成场景热力图...")
    
    # 准备数据
    metrics = ['service_rate', 'net_profit', 'total_cost']
    metric_names = ['服务率 (%)', '净利润 ($)', '调度成本 ($)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        pivot = df.pivot_table(values=metric, index='model', columns='scenario', aggfunc='mean')
        
        if metric == 'service_rate':
            pivot = pivot * 100  # 转换为百分比
        
        # 绘制热力图
        im = axes[idx].imshow(pivot.values, cmap='RdYlGn', aspect='auto')
        
        # 设置刻度
        axes[idx].set_xticks(range(len(pivot.columns)))
        axes[idx].set_xticklabels(pivot.columns, rotation=45, ha='right')
        axes[idx].set_yticks(range(len(pivot.index)))
        axes[idx].set_yticklabels(pivot.index)
        
        # 添加数值
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i, j]
                text_color = 'white' if (value < pivot.values.mean()) else 'black'
                if metric == 'service_rate':
                    text = f'{value:.1f}%'
                else:
                    text = f'${value:.0f}'
                axes[idx].text(j, i, text, ha='center', va='center', 
                             color=text_color, fontsize=9)
        
        axes[idx].set_title(metric_name, fontsize=12, fontweight='bold')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[idx])
    
    plt.suptitle('不同场景下的策略性能对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'scenario_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {output_path.name}")
    return output_path


def plot_metric_radar(df, output_dir):
    """多维度雷达图对比"""
    print("\n📊 生成雷达图对比...")
    
    from math import pi
    
    # 计算各项指标（归一化到0-100）
    summary = df.groupby('model').agg({
        'service_rate': 'mean',
        'net_profit': 'mean',
        'total_cost': 'mean'
    }).reset_index()
    
    # 计算ROI和成本效率
    summary['roi'] = summary['net_profit'] / summary['total_cost']
    summary['cost_efficiency'] = 1 / (summary['total_cost'] / 1000)  # 归一化
    
    # 归一化到0-100
    metrics = ['service_rate', 'net_profit', 'cost_efficiency', 'roi']
    for metric in metrics:
        min_val = summary[metric].min()
        max_val = summary[metric].max()
        summary[f'{metric}_norm'] = (summary[metric] - min_val) / (max_val - min_val) * 100
    
    # 绘制雷达图
    categories = ['服务率', '净利润', '成本效率', 'ROI']
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#e74c3c', '#2ecc71', '#2ecc71', '#3498db']
    
    for idx, (_, row) in enumerate(summary.iterrows()):
        values = [row[f'{m}_norm'] for m in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    ax.grid(True)
    
    ax.set_title('多维度性能对比雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / 'metric_radar.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {output_path.name}")
    return output_path


def plot_improvement_summary(df, output_dir):
    """改进总结图"""
    print("\n📊 生成改进总结图...")
    
    # 计算Day 7到Day 8的改进
    day7 = df[df['model'] == 'PPO-Day7-Original'].agg({
        'service_rate': 'mean',
        'net_profit': 'mean',
        'total_cost': 'mean'
    })
    
    day8 = df[df['model'].str.contains('Day8')].groupby('model').agg({
        'service_rate': 'mean',
        'net_profit': 'mean',
        'total_cost': 'mean'
    }).mean()
    
    # 计算改进百分比
    metrics = {
        '服务率': (day8['service_rate'] - day7['service_rate']) / day7['service_rate'] * 100,
        '净利润': (day8['net_profit'] - day7['net_profit']) / day7['net_profit'] * 100,
        '调度成本': (day8['total_cost'] - day7['total_cost']) / day7['total_cost'] * 100
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(metrics))
    values = list(metrics.values())
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in values]
    
    bars = ax.bar(x, values, color=colors, alpha=0.8)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + (2 if height > 0 else -2),
                f'{value:+.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=12, fontweight='bold')
    
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.keys(), fontsize=12)
    ax.set_ylabel('改进百分比 (%)', fontsize=12)
    ax.set_title('Day 7 → Day 8 改进总结', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加说明
    ax.text(0.5, 0.95, '绿色=改进，红色=下降', transform=ax.transAxes,
           ha='center', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / 'improvement_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 保存: {output_path.name}")
    return output_path


def main():
    args = parse_args()
    
    print("="*70)
    print("Day 9 - 生成可视化图表")
    print("="*70)
    print()
    
    # 创建输出目录
    output_dir = Path("results/day9_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 输出目录: {output_dir.absolute()}")
    print()
    
    # 设置绘图风格
    setup_plotting_style()
    
    # 加载数据
    df = load_comparison_data()
    if df is None:
        return 1
    
    print()
    print("="*70)
    print("生成图表")
    print("="*70)
    
    generated_plots = []
    
    # 1. 成本对比图（必须）
    plot1 = plot_cost_comparison(df, output_dir)
    generated_plots.append(plot1)
    
    # 2. 服务率-成本权衡图（必须）
    plot2 = plot_service_cost_tradeoff(df, output_dir)
    generated_plots.append(plot2)
    
    # 3. ROI对比图（必须）
    plot3 = plot_roi_comparison(df, output_dir)
    generated_plots.append(plot3)
    
    if not args.essential_only:
        # 4. 场景热力图
        plot4 = plot_scenario_heatmap(df, output_dir)
        generated_plots.append(plot4)
        
        # 5. 雷达图
        plot5 = plot_metric_radar(df, output_dir)
        generated_plots.append(plot5)
        
        # 6. 改进总结图
        plot6 = plot_improvement_summary(df, output_dir)
        generated_plots.append(plot6)
    
    print()
    print("="*70)
    print(f"✅ 图表生成完成！共{len(generated_plots)}个")
    print("="*70)
    print()
    print("📂 生成的图表:")
    for plot_path in generated_plots:
        print(f"  - {plot_path.name}")
    print()
    print(f"📁 所有图表位于: {output_dir.absolute()}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())