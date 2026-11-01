"""唯一默认配置：当前选定SwiGLU+RoPE模型、data与E05。"""
import json
from project_paths import DEFAULT_CONFIG, project_path


def _settings():
    return json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))


def generation_defaults():
    return _settings()['generation']


def default_config():
    settings = _settings()
    return {key: settings[key] for key in ('model', 'training', 'data', 'seed')}


def load_config(experiment_id='minigpt_v2', path=None, seed=None):
    # experiment_id只命名运行，不再隐式切换消融架构。
    cfg = default_config()
    if path:
        override = json.loads(project_path(path).read_text(encoding='utf-8'))
        for key, value in override.items():
            if key == 'generation':
                continue  # 同一配置文件里的生成设置不参与训练/续训身份。
            if key not in cfg:
                raise ValueError(f'未知配置项: {key}')
            if isinstance(cfg[key], dict):
                if not isinstance(value, dict) or value.keys() - cfg[key].keys():
                    raise ValueError(f'未知配置字段或格式错误: {key}')
                cfg[key].update(value)
            else:
                cfg[key] = value
    if seed is not None:
        cfg['seed'] = seed
    return cfg
