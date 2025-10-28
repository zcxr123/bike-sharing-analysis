# Day 6 可视化分析报告

**生成时间**: 2025-10-28 21:02:57

## 一、策略对比摘要

### 1.1 整体性能

```
                policy  service_rate_mean  service_rate_std  net_profit_mean  net_profit_std  rebalance_cost_mean  rebalance_cost_std
         Min-Cost-Flow           0.954161          0.005955    120212.938312     8802.402415             0.000000            0.000000
Proportional-Optimized           0.999938          0.000058    125149.235434     9677.148582          1146.989373           75.165349
           Zero-Action           0.950485          0.014135    119978.351497     7798.340711             0.000000            0.000000
```

### 1.2 关键发现

1. **Proportional-Optimized策略表现最佳**
   - 服务率: 1.0% ± 0.0%
   - 净利润: $125149 ± $9677
   - 调度成本: $1147 ± $75

## 二、可视化文件

- `visualizations/policy_comparison.png` - 策略对比柱状图
- `visualizations/scenario_analysis.png` - 场景敏感性分析
- `detailed_comparison_table.csv` - 详细对比表

## 三、下一步建议

1. ✅ **M2阶段完成** - 基线策略评估完成
2. 🎯 **进入M3阶段** - 强化学习训练（PPO/DQN）
3. 📊 **使用最佳基线作为RL对比基准**
4. 🔬 **探索RL是否能进一步提升性能**

---

*报告生成时间: 2025-10-28 21:02:57*
