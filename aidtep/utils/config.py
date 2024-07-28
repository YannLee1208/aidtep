import yaml
import os
from loguru import logger

from aidtep.utils.file import check_file_exist


class AidtepConfig:
    """
    AidtepConfig class to load and store the configuration.
    Usage:
    ```
    config = AidtepConfig('config.yaml')
    value = config.get('key1.key2.key3', default=None)
    ```
    """

    def __init__(self, config_path: str = None, config_dict: dict = None):
        self.config_path = config_path
        if config_dict is not None:
            self.config = self._wrap_config(config_dict)
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

    def _load_config(self, config_path: str) -> dict:
        if not check_file_exist(config_path):
            logger.error(f"Config file '{config_path}' not found")
            raise FileNotFoundError(f"Config file '{config_path}' not found")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return self._wrap_config(config_dict)

    def _wrap_config(self, config_dict: dict):
        """
        Recursively wrap a dictionary into AidtepConfig objects.
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                config_dict[key] = AidtepConfig(config_dict=value)
        return config_dict

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
        except KeyError:
            return default
        return value

    def keys(self) -> list:
        return list(self.config.keys())

    def __getitem__(self, key: str):
        if key not in self.keys():
            raise KeyError(f"Key '{key}' not found in config")
        return self.get(key)

    def __repr__(self):
        return f"AidtepConfig({self.config})"


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", 'config', 'dev.yaml')
    config = AidtepConfig(config_path)
    print(config)

    processes = config.get("data_process")
    print(processes)
    print(processes.keys())
    print(config.get("data_process.IAEA.use"))
