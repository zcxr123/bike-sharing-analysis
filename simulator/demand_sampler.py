"""
需求采样模块 (Demand Sampler)
基于λ(t)参数进行泊松采样，生成各区域的需求量

Author: renr
Date: 2025-10-28
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class DemandSampler:
    """
    需求采样器
    
    基于预训练的λ(t)参数，结合时间、天气、季节等上下文信息，
    使用泊松分布采样各区域的需求量。
    """
    
    def __init__(
        self,
        lambda_params_path: str,
        zone_weights: List[float],
        demand_scale: float = 1.0,
        random_seed: Optional[int] = None
    ):
        """
        初始化需求采样器
        
        Args:
            lambda_params_path: lambda参数文件路径（pkl格式）
            zone_weights: 各区域权重列表，和为1
            demand_scale: 需求放大系数
            random_seed: 随机种子（可选）
        """
        self.zone_weights = np.array(zone_weights)
        self.demand_scale = demand_scale
        self.num_zones = len(zone_weights)
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 加载lambda参数
        self.lambda_params = self._load_lambda_params(lambda_params_path)
        
        # 提取参数字典
        self.hourly_lambda = self.lambda_params.get('hourly', {})
        self.seasonal_lambda = self.lambda_params.get('seasonal', {})
        self.weekday_lambda = self.lambda_params.get('weekday', {})
        self.workingday_lambda = self.lambda_params.get('workingday', {})
        self.weather_lambda = self.lambda_params.get('weather', {})
        self.overall_mean = self.lambda_params.get('overall_mean', 189.5)
        self.overall_std = self.lambda_params.get('overall_std', 181.4)
        
        print(f"[DemandSampler] 初始化完成")
        print(f"  - 区域数量: {self.num_zones}")
        print(f"  - 需求放大系数: {self.demand_scale}")
        print(f"  - 总体平均需求: {self.overall_mean:.2f} 单/小时")
    
    def _load_lambda_params(self, path):
        """
        更鲁棒的 Lambda 参数加载：
        - 依次尝试：传入路径（按当前工作目录解析）、以项目根为基准的相对路径、
          project_root/results/<basename>、并搜索 project_root/results 下的 pkl 候选文件。
        - 若都不存在，则生成默认占位参数并保存到 project_root/results/<basename>
        """
        import pickle
        import numpy as np
        from pathlib import Path

        proj_root = Path(__file__).parent.parent  # 项目根
        p = Path(path)

        # 构建候选路径（按尝试顺序）
        candidates = []
        # 如果传入的是绝对路径，优先尝试
        if p.is_absolute():
            candidates.append(p)
        # 传入路径（相对当前工作目录）
        candidates.append(p)
        # 相对于项目根的同一路径（处理 "../results/..." 这类情况）
        candidates.append(proj_root / p)
        # 直接在 project_root/results 下查找同名文件
        candidates.append(proj_root / "results" / p.name)

        # 去重并尝试加载
        seen = set()
        for cand in candidates:
            try:
                cand_resolved = cand.resolve()
            except Exception:
                cand_resolved = cand
            if str(cand_resolved) in seen:
                continue
            seen.add(str(cand_resolved))
            try:
                if cand.exists():
                    with cand.open("rb") as f:
                        obj = pickle.load(f)
                    print(f"[Info] 从 {cand.resolve()} 加载 Lambda 参数")
                    return obj
            except Exception as e:
                print(f"[Warning] 读取 Lambda 参数文件失败 ({cand}): {e}")

        # 作为最后手段，扫描 project_root/results 下的 pkl 文件，尝试加载第一个可读的
        results_dir = proj_root / "results"
        if results_dir.exists():
            for cand in results_dir.glob("*.pkl"):
                try:
                    with cand.open("rb") as f:
                        obj = pickle.load(f)
                    print(f"[Info] 从候选文件 {cand.resolve()} 加载 Lambda 参数")
                    return obj
                except Exception:
                    continue

        # 都没有 -> 生成默认参数并保存到 project_root/results/<basename>
        print(f"[Warning] 未找到 Lambda 参数文件（尝试路径：{', '.join([str(c) for c in candidates])}）。将生成默认参数并保存到 {results_dir.resolve()}")
        default_params = {"default": np.ones(24) * 100}
        out_path = results_dir / (p.name or "lambda_params.pkl")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as f:
                pickle.dump(default_params, f)
            print(f"[Info] 已写入默认 Lambda 参数: {out_path.resolve()}")
        except Exception as e:
            print(f"[Error] 无法写入默认 Lambda 参数文件 {out_path}: {e}")

        return default_params
    
    def get_lambda_t(
        self,
        hour: int,
        season: int,
        workingday: int,
        weather: int,
        use_mean: bool = False
    ) -> float:
        """
        获取时刻t的需求强度λ_t
        
        Args:
            hour: 小时 (0-23)
            season: 季节 (1-4)
            workingday: 是否工作日 (0/1)
            weather: 天气类型 (1-4)
            use_mean: 是否使用简单平均（不使用模型预测）
        
        Returns:
            lambda_t: 需求强度（订单/小时）
        """
        if use_mean:
            # 简单平均法（基线）
            lambda_t = self.overall_mean
        else:
            # 基于各维度参数的组合（加权平均）
            lambda_hour = self.hourly_lambda.get(hour, self.overall_mean)
            lambda_season = self.seasonal_lambda.get(season, self.overall_mean)
            lambda_workingday = self.workingday_lambda.get(workingday, self.overall_mean)
            lambda_weather = self.weather_lambda.get(weather, self.overall_mean)
            
            # 组合方式：加权平均（可以改为其他方式）
            # 权重：小时40%，季节20%，工作日20%，天气20%
            lambda_t = (
                0.4 * lambda_hour +
                0.2 * lambda_season +
                0.2 * lambda_workingday +
                0.2 * lambda_weather
            )
        
        # 应用放大系数
        lambda_t *= self.demand_scale
        
        return max(0.0, lambda_t)  # 确保非负
    
    def sample_demand(
        self,
        hour: int,
        season: int,
        workingday: int,
        weather: int,
        extreme_event: bool = False,
        extreme_multiplier: float = 2.0
    ) -> np.ndarray:
        """
        采样各区域的需求量
        
        Args:
            hour: 当前小时 (0-23)
            season: 季节 (1-4)
            workingday: 是否工作日 (0/1)
            weather: 天气类型 (1-4)
            extreme_event: 是否为极端事件（高需求场景）
            extreme_multiplier: 极端事件需求倍增系数
        
        Returns:
            demands: 各区域需求量数组 (num_zones,)
        """
        # 获取整体需求强度
        lambda_t = self.get_lambda_t(hour, season, workingday, weather)
        
        # 极端事件调整
        if extreme_event:
            lambda_t *= extreme_multiplier
        
        # 按区域权重分配需求强度
        lambda_zones = lambda_t * self.zone_weights  # (num_zones,)
        
        # 泊松采样
        demands = np.random.poisson(lambda_zones)
        
        return demands.astype(np.float32)
    
    def sample_batch_demands(
        self,
        hours: np.ndarray,
        seasons: np.ndarray,
        workingdays: np.ndarray,
        weathers: np.ndarray,
        extreme_events: Optional[np.ndarray] = None,
        extreme_multiplier: float = 2.0
    ) -> np.ndarray:
        """
        批量采样需求（向量化版本，用于训练加速）
        
        Args:
            hours: 小时数组 (batch_size,)
            seasons: 季节数组 (batch_size,)
            workingdays: 工作日数组 (batch_size,)
            weathers: 天气数组 (batch_size,)
            extreme_events: 极端事件标识 (batch_size,) 可选
            extreme_multiplier: 极端事件倍增系数
        
        Returns:
            demands: 需求数组 (batch_size, num_zones)
        """
        batch_size = len(hours)
        demands = np.zeros((batch_size, self.num_zones), dtype=np.float32)
        
        for i in range(batch_size):
            is_extreme = extreme_events[i] if extreme_events is not None else False
            demands[i] = self.sample_demand(
                int(hours[i]),
                int(seasons[i]),
                int(workingdays[i]),
                int(weathers[i]),
                extreme_event=bool(is_extreme),
                extreme_multiplier=extreme_multiplier
            )
        
        return demands
    
    def get_expected_demand(
        self,
        hour: int,
        season: int,
        workingday: int,
        weather: int
    ) -> np.ndarray:
        """
        获取期望需求（不采样，返回λ_t本身）
        
        用于策略评估和分析
        
        Returns:
            expected_demands: 各区域期望需求 (num_zones,)
        """
        lambda_t = self.get_lambda_t(hour, season, workingday, weather)
        expected_demands = lambda_t * self.zone_weights
        return expected_demands.astype(np.float32)
    
    def get_demand_statistics(self, num_samples: int = 1000) -> Dict:
        """
        获取需求统计信息（用于分析和调试）
        
        Args:
            num_samples: 采样次数
        
        Returns:
            stats: 统计信息字典
        """
        # 随机采样不同场景
        hours = np.random.randint(0, 24, num_samples)
        seasons = np.random.randint(1, 5, num_samples)
        workingdays = np.random.randint(0, 2, num_samples)
        weathers = np.random.randint(1, 5, num_samples)
        
        # 采样需求
        demands = self.sample_batch_demands(hours, seasons, workingdays, weathers)
        
        # 计算统计量
        stats = {
            'mean': demands.mean(),
            'std': demands.std(),
            'min': demands.min(),
            'max': demands.max(),
            'median': np.median(demands),
            'total_mean': demands.sum(axis=1).mean(),  # 全城总需求均值
            'zone_means': demands.mean(axis=0).tolist(),  # 各区域均值
            'zone_stds': demands.std(axis=0).tolist(),    # 各区域标准差
        }
        
        return stats


# ==================== 辅助函数 ====================

def create_demand_sampler_from_config(config: Dict) -> DemandSampler:
    """
    从配置字典创建需求采样器
    
    Args:
        config: 配置字典（从yaml加载）
    
    Returns:
        sampler: DemandSampler实例
    """
    return DemandSampler(
        lambda_params_path=config['demand']['lambda_params_path'],
        zone_weights=config['zones']['zone_weights'],
        demand_scale=config['demand']['demand_scale'],
        random_seed=config['demand']['random_seed']
    )


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试需求采样器
    print("="*60)
    print("需求采样器测试")
    print("="*60)
    
    # 假设lambda_params.pkl在项目根目录的results文件夹下
    # 如果没有，需要先运行Day2的需求模型拟合代码生成
    
    try:
        sampler = DemandSampler(
            lambda_params_path="../results/lambda_params.pkl",
            zone_weights=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10],
            demand_scale=1.0,
            random_seed=42
        )
        
        # 测试1: 单次采样
        print("\n[测试1] 单次需求采样")
        print("-" * 40)
        hour, season, workingday, weather = 17, 3, 1, 1  # 夏季晴天工作日17:00
        demands = sampler.sample_demand(hour, season, workingday, weather)
        print(f"场景: 夏季晴天工作日 17:00")
        print(f"各区域需求: {demands}")
        print(f"总需求: {demands.sum():.0f} 单")
        
        # 测试2: 期望需求
        print("\n[测试2] 期望需求")
        print("-" * 40)
        expected = sampler.get_expected_demand(hour, season, workingday, weather)
        print(f"各区域期望需求: {expected}")
        print(f"期望总需求: {expected.sum():.2f} 单")
        
        # 测试3: 批量采样
        print("\n[测试3] 批量采样")
        print("-" * 40)
        batch_size = 10
        hours = np.array([8, 12, 17, 20, 23] * 2)
        seasons = np.array([2] * batch_size)
        workingdays = np.array([1] * batch_size)
        weathers = np.array([1] * batch_size)
        
        batch_demands = sampler.sample_batch_demands(hours, seasons, workingdays, weathers)
        print(f"批量大小: {batch_size}")
        print(f"平均总需求: {batch_demands.sum(axis=1).mean():.2f} 单")
        
        # 测试4: 统计信息
        print("\n[测试4] 需求统计信息")
        print("-" * 40)
        stats = sampler.get_demand_statistics(num_samples=1000)
        print(f"总需求均值: {stats['total_mean']:.2f}")
        print(f"总需求标准差: {stats['std']:.2f}")
        print(f"需求范围: [{stats['min']:.0f}, {stats['max']:.0f}]")
        
        print("\n✅ 需求采样器测试通过！")
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("提示: 请先运行Day2的需求模型代码生成lambda_params.pkl文件")