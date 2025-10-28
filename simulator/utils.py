import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent

def load_config(path: str = None) -> dict:
    """
    加载配置，支持多种默认路径：
    1) 显式 path
    2) config/config.yaml
    3) config/env_config.yaml
    4) config.yaml (项目根)
    """
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates += [
        project_root / 'config' / 'config.yaml',
        project_root / 'config' / 'env_config.yaml',
        project_root / 'config.yaml'
    ]

    for p in candidates:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)

    # 未找到任何配置文件，抛出清晰错误
    raise FileNotFoundError(
        f"No config file found. Searched: {', '.join(str(p) for p in candidates)}"
    )
