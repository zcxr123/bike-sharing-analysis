"""
测试报告生成功能
"""
import sys
sys.path.insert(0, '..')

import pandas as pd
from datetime import datetime
from pathlib import Path

# 模拟数据
comparison_data = [
    {
        '策略': 'Zero-Action',
        '服务率(%)': '89.94 ± 2.60',
        '净利润($)': '92116.33 ± 1345.23',
        '调度成本($)': '0.00 ± 0.00',
        '未满足需求': '4115.8',
        '总服务量': 150657,
        '总未满足': 20579
    }
]

comparison_df = pd.DataFrame(comparison_data)

results = [{
    'policy_name': 'Zero-Action',
    'num_episodes': 5,
    'mean_service_rate': 0.8994,
    'std_service_rate': 0.0260,
    'mean_net_profit': 92116.33,
    'std_net_profit': 1345.23,
    'mean_rebalance_cost': 0.0,
    'std_rebalance_cost': 0.0,
    'mean_unmet_demand': 4115.8,
    'total_served': 150657,
    'total_unmet': 20579
}]

# 加载配置
import yaml
with open('../config/env_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 测试生成报告
output_dir = Path('../results')
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_file = output_dir / f"test_report_{timestamp}.md"

print(f"尝试生成报告到: {report_file}")
print(f"文件路径是否存在: {output_dir.exists()}")
print(f"是否有写入权限: {output_dir.stat().st_mode}")

try:
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 测试报告\n\n")
        f.write("这是一个测试报告。\n")
    
    print(f"✅ 测试报告生成成功！")
    print(f"文件大小: {report_file.stat().st_size} bytes")
    print(f"文件内容:")
    with open(report_file, 'r', encoding='utf-8') as f:
        print(f.read())
    
except Exception as e:
    print(f"❌ 报告生成失败: {e}")
    import traceback
    traceback.print_exc()
