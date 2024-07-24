import yaml
import os
from loguru import logger


class AidtepConfig:
    """
    AidtepConfig is a singleton class to load and store the configuration.
    Usage:
    ```
    init_config('config.yaml')
    config = get_config()
    config.get('key1.key2.key3', default=None)
    ```
    """
    _instance = None

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(AidtepConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str = None):
        if self._initialized:
            logger.warning("config already initialized, skipping")
            return
        self.config_path = config_path
        self.config = self._load_config()
        self._initialized = True

    def _load_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
        except KeyError:
            return default
        return value

    def __getitem__(self, key: str):
        return self.get(key)

    def __repr__(self):
        return f"Config({self.config})"


def init_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return AidtepConfig(config_path)


# 获取全局配置实例
def get_config():
    if AidtepConfig._instance is None or not AidtepConfig._instance._initialized:
        logger.error("Config is not initialized, call init_config(config_path) first.")
        raise ValueError("Config is not initialized, call init_config(config_path) first.")
    return AidtepConfig._instance
