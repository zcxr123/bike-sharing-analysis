#!/usr/bin/env python3
"""
Day 6: 深入分析与可视化 - 自适应版本
自动检测CSV列名并适配
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class Day6Visualizer:
    """Day6可视化器 - 自适应版本"""
    
    # 列名映射（支持多种可能的列名）
    COLUMN_MAPPINGS = {
        'service_rate': ['service_rate', 'served_rate', 'fulfillment_rate'],
        'net_profit': ['net_profit', 'profit', 'total_profit'],
        'rebalance_cost': ['rebalance_cost', 'total_cost', 'cost', 'rebalancing_cost'],
        'unmet_demand': ['unmet_demand', 'unmet', 'shortage', 'unfulfilled_demand']
    }
    
    def __init__(self, results_dir='results'):
        """初始化可视化器"""
        self.results_dir = Path(results_dir)
        
        # 智能路径处理
        if not self.results_dir.exists():
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            alt_results_dir = project_root / 'results'
            
            if alt_results_dir.exists():
                self.results_dir = alt_results_dir
                print(f"📂 找到results目录: {self.results_dir.absolute()}")
            else:
                print(f"⚠️  警告: results目录不存在，创建中...")
                self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz_dir = self.results_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("Day 6: 深入分析与可视化")
        print("="*70)
        print(f"📁 Results目录: {self.results_dir.absolute()}")
        print(f"📁 可视化输出: {self.viz_dir.absolute()}")
        
        # 存储列名映射
        self.actual_columns = {}
    
    def detect_column_names(self, df):
        """智能检测实际的列名"""
        print("\n🔍 检测列名...")
        
        available_cols = df.columns.tolist()
        print(f"   可用列: {available_cols}")
        
        for standard_name, possible_names in self.COLUMN_MAPPINGS.items():
            found = False
            for possible_name in possible_names:
                if possible_name in available_cols:
                    self.actual_columns[standard_name] = possible_name
                    found = True
                    break
            
            if not found:
                print(f"   ⚠️  警告: 未找到 {standard_name} 列")
                self.actual_columns[standard_name] = None
        
        print(f"   ✅ 列名映射: {self.actual_columns}")
        return self.actual_columns
    
    def load_results(self):
        """加载Day5的评估结果"""
        print("\n📁 加载评估结果...")
        
        # 找到最新的结果文件
        comparison_files = sorted(self.results_dir.glob('multi_scenario_comparison_*.csv'))
        
        if not comparison_files:
            print(f"❌ 未找到评估结果文件！")
            print(f"   搜索路径: {self.results_dir.absolute()}")
            
            # 列出所有CSV文件
            all_csvs = list(self.results_dir.glob('*.csv'))
            if all_csvs:
                print(f"\n   找到的CSV文件:")
                for csv_file in all_csvs:
                    print(f"   - {csv_file.name}")
            
            return None
        
        latest_file = comparison_files[-1]
        df = pd.read_csv(latest_file)
        
        print(f"✅ 加载成功: {latest_file.name}")
        print(f"   包含 {len(df)} 条记录")
        print(f"   策略: {df['policy'].unique().tolist()}")
        print(f"   场景: {df['scenario'].unique().tolist()}")
        
        # 检测列名
        self.detect_column_names(df)
        
        return df
    
    def get_column(self, standard_name):
        """获取实际的列名"""
        return self.actual_columns.get(standard_name, standard_name)
    
    def plot_policy_comparison(self, df):
        """任务1.1: 策略对比图"""
        print("\n" + "="*70)
        print("任务1: 策略对比可视化")
        print("="*70)
        
        # 构建聚合字典（只包含存在的列）
        agg_dict = {}
        metrics_to_plot = []
        
        for metric in ['service_rate', 'net_profit', 'rebalance_cost']:
            actual_col = self.get_column(metric)
            if actual_col and actual_col in df.columns:
                agg_dict[actual_col] = ['mean', 'std']
                metrics_to_plot.append((metric, actual_col))
        
        if not agg_dict:
            print("❌ 没有可用的指标进行绘图")
            return None
        
        # 按策略聚合
        policy_summary = df.groupby('policy').agg(agg_dict).reset_index()
        
        # 展平列名
        new_columns = ['policy']
        for metric, actual_col in metrics_to_plot:
            new_columns.extend([f'{metric}_mean', f'{metric}_std'])
        policy_summary.columns = new_columns
        
        print("\n策略性能汇总:")
        print(policy_summary.to_string(index=False))
        
        # 创建对比图
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
        if num_metrics == 1:
            axes = [axes]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        metric_labels = {
            'service_rate': ('Service Rate (%)', 0, 105),
            'net_profit': ('Net Profit ($)', None, None),
            'rebalance_cost': ('Rebalance Cost ($)', None, None)
        }
        
        for idx, (metric, actual_col) in enumerate(metrics_to_plot):
            ax = axes[idx]
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            bars = ax.bar(range(len(policy_summary)), 
                         policy_summary[mean_col],
                         yerr=policy_summary[std_col],
                         capsize=5, alpha=0.7, color=colors[:len(policy_summary)])
            
            ax.set_xticks(range(len(policy_summary)))
            ax.set_xticklabels(policy_summary['policy'], rotation=15, ha='right')
            
            ylabel, ymin, ymax = metric_labels[metric]
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{ylabel.split("(")[0].strip()} Comparison', 
                        fontsize=14, fontweight='bold')
            
            if ymin is not None and ymax is not None:
                ax.set_ylim([ymin, ymax])
            
            ax.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if metric == 'service_rate':
                    label = f'{height:.1f}%'
                else:
                    label = f'${height:.0f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存
        output_path = self.viz_dir / 'policy_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 策略对比图已保存: {output_path}")
        plt.close()
        
        return policy_summary
    
    def plot_scenario_analysis(self, df):
        """任务1.2: 场景敏感性分析"""
        print("\n" + "-"*70)
        print("场景敏感性分析")
        print("-"*70)
        
        # 确定可用的指标
        metrics_to_plot = []
        for metric in ['service_rate', 'net_profit']:
            actual_col = self.get_column(metric)
            if actual_col and actual_col in df.columns:
                metrics_to_plot.append((metric, actual_col))
        
        if not metrics_to_plot:
            print("❌ 没有可用的指标进行场景分析")
            return
        
        # 按场景和策略分组
        pivot_dict = {actual_col: 'mean' for _, actual_col in metrics_to_plot}
        scenario_pivot = df.pivot_table(
            index='scenario',
            columns='policy',
            values=[actual_col for _, actual_col in metrics_to_plot],
            aggfunc='mean'
        )
        
        print("\n场景性能汇总:")
        print(scenario_pivot.to_string())
        
        # 绘制分组柱状图
        num_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(1, num_metrics, figsize=(7*num_metrics, 5))
        if num_metrics == 1:
            axes = [axes]
        
        for idx, (metric, actual_col) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            scenario_pivot[actual_col].plot(kind='bar', ax=ax, width=0.8, alpha=0.7)
            
            if metric == 'service_rate':
                ax.set_ylabel('Service Rate (%)', fontsize=12)
                ax.set_title('Service Rate by Scenario', fontsize=14, fontweight='bold')
                ax.set_ylim([0, 105])
            else:
                ax.set_ylabel('Net Profit ($)', fontsize=12)
                ax.set_title('Net Profit by Scenario', fontsize=14, fontweight='bold')
            
            ax.legend(title='Policy', loc='lower right')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存
        output_path = self.viz_dir / 'scenario_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 场景分析图已保存: {output_path}")
        plt.close()
    
    def plot_detailed_comparison_table(self, df):
        """生成详细对比表"""
        print("\n" + "-"*70)
        print("生成详细对比表")
        print("-"*70)
        
        # 确定可用的列
        agg_dict = {}
        for metric in ['service_rate', 'net_profit', 'rebalance_cost', 'unmet_demand']:
            actual_col = self.get_column(metric)
            if actual_col and actual_col in df.columns:
                agg_dict[actual_col] = 'mean'
        
        # 按策略和场景聚合
        detailed = df.groupby(['policy', 'scenario']).agg(agg_dict).round(2)
        
        # 保存为CSV
        output_path = self.results_dir / 'detailed_comparison_table.csv'
        detailed.to_csv(output_path)
        print(f"✅ 详细对比表已保存: {output_path}")
        
        # 打印摘要
        print("\n" + "="*70)
        print("策略性能摘要（所有场景平均）")
        print("="*70)
        
        agg_summary = {}
        for metric in ['service_rate', 'net_profit', 'rebalance_cost']:
            actual_col = self.get_column(metric)
            if actual_col and actual_col in df.columns:
                agg_summary[actual_col] = ['mean', 'std']
        
        if agg_summary:
            summary = df.groupby('policy').agg(agg_summary).round(2)
            print(summary.to_string())
    
    def generate_report(self, policy_summary):
        """生成Day6评估报告"""
        print("\n" + "="*70)
        print("生成评估报告")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f'day6_visualization_report_{timestamp}.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Day 6 可视化分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 一、策略对比摘要\n\n")
            f.write("### 1.1 整体性能\n\n")
            f.write("```\n")
            f.write(policy_summary.to_string(index=False))
            f.write("\n```\n\n")
            
            f.write("### 1.2 关键发现\n\n")
            
            # 找出最佳策略（基于净利润）
            if 'net_profit_mean' in policy_summary.columns:
                best_policy_idx = policy_summary['net_profit_mean'].idxmax()
                best_policy = policy_summary.iloc[best_policy_idx]
                
                f.write(f"1. **{best_policy['policy']}策略表现最佳**\n")
                
                if 'service_rate_mean' in policy_summary.columns:
                    f.write(f"   - 服务率: {best_policy['service_rate_mean']:.1f}%")
                    if 'service_rate_std' in policy_summary.columns:
                        f.write(f" ± {best_policy['service_rate_std']:.1f}%")
                    f.write("\n")
                
                f.write(f"   - 净利润: ${best_policy['net_profit_mean']:.0f}")
                if 'net_profit_std' in policy_summary.columns:
                    f.write(f" ± ${best_policy['net_profit_std']:.0f}")
                f.write("\n")
                
                if 'rebalance_cost_mean' in policy_summary.columns:
                    f.write(f"   - 调度成本: ${best_policy['rebalance_cost_mean']:.0f}")
                    if 'rebalance_cost_std' in policy_summary.columns:
                        f.write(f" ± ${best_policy['rebalance_cost_std']:.0f}")
                    f.write("\n\n")
            
            f.write("## 二、可视化文件\n\n")
            f.write("- `visualizations/policy_comparison.png` - 策略对比柱状图\n")
            f.write("- `visualizations/scenario_analysis.png` - 场景敏感性分析\n")
            f.write("- `detailed_comparison_table.csv` - 详细对比表\n\n")
            
            f.write("## 三、下一步建议\n\n")
            f.write("1. ✅ **M2阶段完成** - 基线策略评估完成\n")
            f.write("2. 🎯 **进入M3阶段** - 强化学习训练（PPO/DQN）\n")
            f.write("3. 📊 **使用最佳基线作为RL对比基准**\n")
            f.write("4. 🔬 **探索RL是否能进一步提升性能**\n\n")
            
            f.write("---\n\n")
            f.write("*报告生成时间: {}*\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        print(f"✅ 报告已保存: {report_path}")
    
    def run_all(self):
        """执行所有可视化任务"""
        # 1. 加载数据
        df = self.load_results()
        if df is None:
            print("\n❌ 无法继续，请确保Day5的评估结果存在")
            return False
        
        # 2. 策略对比
        policy_summary = self.plot_policy_comparison(df)
        if policy_summary is None:
            print("\n❌ 策略对比失败")
            return False
        
        # 3. 场景分析
        self.plot_scenario_analysis(df)
        
        # 4. 详细对比表
        self.plot_detailed_comparison_table(df)
        
        # 5. 生成报告
        self.generate_report(policy_summary)
        
        print("\n" + "="*70)
        print("✅ Day 6 可视化任务完成！")
        print("="*70)
        print(f"📁 输出目录: {self.viz_dir.absolute()}")
        print(f"📝 查看报告以了解详细结果")
        print("\n🎯 下一步: 进入Day 7 - RL训练（M3阶段）")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Day 6 可视化与分析')
    parser.add_argument('--results-dir', default='results', 
                       help='结果目录路径（默认: results）')
    
    args = parser.parse_args()
    
    visualizer = Day6Visualizer(results_dir=args.results_dir)
    success = visualizer.run_all()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()