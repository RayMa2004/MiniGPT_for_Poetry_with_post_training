"""主链路路径集中定义；相对CLI路径统一相对于项目根目录。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / 'configs/minigpt_v2.json'
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'checkpoints/best.pt'
DEFAULT_VOCAB = PROJECT_ROOT / 'data/vocab.json'
DEFAULT_PROMPTS = PROJECT_ROOT / 'evaluation/prompts.json'
DEFAULT_RUNS = PROJECT_ROOT / 'runs'
DEFAULT_EVALUATIONS = PROJECT_ROOT / 'evaluations'


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
