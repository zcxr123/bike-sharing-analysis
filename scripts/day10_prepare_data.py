#!/usr/bin/env python3
"""
Day 10 - 数据准备脚本
为Dashboard准备所有需要的数据
"""

import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def prepare_dashboard_data():
    """准备Dashboard所需的所有数据"""
    print("="*70)
    print("Day 10 - 准备Dashboard数据")
    print("="*70)
    print()
    
    # 创建输出目录
    dashboard_dir = project_root / "dashboard"
    data_dir = dashboard_dir / "data"
    assets_dir = dashboard_dir / "assets" / "plots"
    
    for dir_path in [data_dir, assets_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Dashboard目录: {dashboard_dir.absolute()}")
    print()
    
    # 1. 加载对比数据
    print("1️⃣  加载策略对比数据...")
    comparison_dir = project_root / "results" / "day8_comparison"
    comparison_files = list(comparison_dir.glob("comparison_detail_*.csv"))
    
    if not comparison_files:
        print("❌ 错误: 找不到对比数据，请先完成Day 8")
        return False
    
    latest_comparison = max(comparison_files, key=lambda p: p.stat().st_mtime)
    comparison_df = pd.read_csv(latest_comparison)
    
    # 保存到dashboard
    comparison_df.to_csv(data_dir / "comparison.csv", index=False)
    print(f"✅ 对比数据: {len(comparison_df)}条记录")
    print()
    
    # 2. 加载决策数据
    print("2️⃣  加载决策分析数据...")
    analysis_dir = project_root / "results" / "day9_analysis"
    decision_files = list(analysis_dir.glob("decision_data_*.csv"))
    
    if not decision_files:
        print("⚠️  警告: 找不到决策数据，将跳过决策分析页面")
        decision_df = None
    else:
        latest_decision = max(decision_files, key=lambda p: p.stat().st_mtime)
        decision_df = pd.read_csv(latest_decision)
        decision_df.to_csv(data_dir / "decisions.csv", index=False)
        print(f"✅ 决策数据: {len(decision_df)}条记录")
    print()
    
    # 3. 生成汇总统计
    print("3️⃣  生成汇总统计...")
    
    summary = {}
    
    # 按模型汇总
    summary['by_model'] = comparison_df.groupby('model').agg({
        'service_rate': ['mean', 'std'],
        'net_profit': ['mean', 'std'],
        'total_cost': ['mean', 'std']
    }).round(2)
    
    # 按场景汇总
    summary['by_scenario'] = comparison_df.groupby('scenario').agg({
        'service_rate': 'mean',
        'net_profit': 'mean',
        'total_cost': 'mean'
    }).round(2)
    
    # 核心指标
    day7_data = comparison_df[comparison_df['model'] == 'PPO-Day7-Original']
    day8_data = comparison_df[comparison_df['model'].str.contains('Day8')]
    baseline_data = comparison_df[comparison_df['model'] == 'Proportional-Optimized']
    
    if len(day7_data) > 0 and len(day8_data) > 0:
        day7_cost = day7_data['total_cost'].mean()
        day8_cost = day8_data['total_cost'].mean()
        day7_profit = day7_data['net_profit'].mean()
        day8_profit = day8_data['net_profit'].mean()
        
        # ROI计算
        day7_roi = day7_profit / day7_cost if day7_cost > 0 else 0
        day8_roi = day8_profit / day8_cost if day8_cost > 0 else 0
        
        summary['core_metrics'] = {
            'cost_reduction_pct': (1 - day8_cost / day7_cost) * 100 if day7_cost > 0 else 0,
            'cost_reduction_abs': day7_cost - day8_cost,
            'roi_day7': day7_roi,
            'roi_day8': day8_roi,
            'roi_improvement': day8_roi / day7_roi if day7_roi > 0 else 0,
            'service_rate_day8': day8_data['service_rate'].mean() * 100,
            'annual_benefit': (day7_cost - day8_cost + day8_profit - day7_profit) * 52
        }
        
        print(f"✅ 核心指标计算完成")
        print(f"   成本降低: {summary['core_metrics']['cost_reduction_pct']:.1f}%")
        print(f"   ROI提升: {summary['core_metrics']['roi_improvement']:.2f}x")
        print(f"   年度效益: ${summary['core_metrics']['annual_benefit']:,.0f}")
    else:
        print("⚠️  警告: 缺少Day 7或Day 8数据，核心指标计算不完整")
        summary['core_metrics'] = {}
    
    print()
    
    # 4. 决策分析统计
    if decision_df is not None:
        print("4️⃣  生成决策分析统计...")
        
        summary['decision_stats'] = {
            'total_decisions': len(decision_df),
            'avg_rebalance_cost': decision_df['rebalance_cost'].mean(),
            'avg_num_moves': decision_df['num_moves'].mean(),
            'total_served': decision_df['total_served'].sum(),
            'total_cost': decision_df['rebalance_cost'].sum()
        }
        
        # 时间模式
        hourly_cost = decision_df.groupby('hour')['rebalance_cost'].sum()
        summary['hourly_pattern'] = {
            'peak_hours': hourly_cost.nlargest(5).index.tolist(),
            'low_hours': hourly_cost.nsmallest(5).index.tolist(),
            'hourly_cost': hourly_cost.to_dict()
        }
        
        print(f"✅ 决策统计完成")
        print(f"   总决策数: {summary['decision_stats']['total_decisions']}")
        print(f"   平均成本: ${summary['decision_stats']['avg_rebalance_cost']:.2f}")
    
    print()
    
    # 5. 保存汇总数据
    print("5️⃣  保存汇总数据...")
    
    with open(data_dir / "summary.pkl", 'wb') as f:
        pickle.dump(summary, f)
    
    # 同时保存为JSON（便于调试）
    import json
    
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            # 处理 MultiIndex 列名，转为可序列化的字符串列名
            df = obj.copy()
            new_cols = []
            for c in df.columns:
                if isinstance(c, tuple):
                    # 过滤 None 并用下划线连接
                    new_cols.append("_".join([str(x) for x in c if x is not None]))
                else:
                    new_cols.append(str(c))
            df.columns = new_cols
            # 返回列 -> 列表 的形式，避免嵌套 tuple 作为 key
            return df.to_dict(orient="list")
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # 强制将非基本类型键转换为字符串
                if not isinstance(k, (str, int, float, bool, type(None))):
                    key = str(k)
                else:
                    key = k
                out[key] = convert_to_serializable(v)
            return out
        if isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    summary_json = convert_to_serializable(summary)
    
    with open(data_dir / "summary.json", 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"✅ 汇总数据已保存")
    print()
    
    # 6. 复制可视化图表
    print("6️⃣  复制可视化图表...")
    
    viz_dir = project_root / "results" / "day9_visualizations"
    if viz_dir.exists():
        import shutil
        plot_files = list(viz_dir.glob("*.png"))
        
        for plot_file in plot_files:
            dest = assets_dir / plot_file.name
            shutil.copy2(plot_file, dest)
        
        print(f"✅ 复制了{len(plot_files)}个图表")
    else:
        print("⚠️  警告: 找不到可视化图表目录")
    
    print()
    
    # 7. 生成配置文件
    print("7️⃣  生成Dashboard配置...")
    
    config = {
        'title': '共享单车智能调度系统',
        'subtitle': '基于强化学习的成本优化方案',
        'version': '1.0',
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_files': {
            'comparison': 'data/comparison.csv',
            'decisions': 'data/decisions.csv' if decision_df is not None else None,
            'summary': 'data/summary.pkl'
        },
        'plot_files': [f.name for f in assets_dir.glob("*.png")]
    }
    
    with open(dashboard_dir / "config.json", 'w') as f:
         json.dump(config, f, indent=2)
    
    print(f"✅ 配置文件已生成")
    print()
    
    # 8. 总结
    print("="*70)
    print("✅ Dashboard数据准备完成！")
    print("="*70)
    print()
    print("📂 输出文件:")
    print(f"  - {data_dir / 'comparison.csv'}")
    if decision_df is not None:
        print(f"  - {data_dir / 'decisions.csv'}")
    print(f"  - {data_dir / 'summary.pkl'}")
    print(f"  - {data_dir / 'summary.json'}")
    print(f"  - {dashboard_dir / 'config.json'}")
    print(f"  - {len(list(assets_dir.glob('*.png')))}个图表")
    print()
    print("🚀 下一步:")
    print("  cd dashboard")
    print("  streamlit run app.py")
    print()
    
    return True


def main():
    try:
        success = prepare_dashboard_data()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())