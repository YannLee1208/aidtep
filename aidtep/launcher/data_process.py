import os

from aidtep.utils.config import get_config
from aidtep.utils.initialize import initialize

config_path = os.path.join(os.path.dirname(__file__), "..", "..", 'config', 'dev.yaml')

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", 'config', 'dev.yaml')
    initialize(config_path)

    config = get_config()
    from aidtep.data_process.IAEA_process import main_process
    main_process(data_path=config.get("data_process.IAEA.input.phione_path"),
                 data_type=config.get("data_process.IAEA.input.data_type", default="float16"),
                 down_sample_factor=config.get("data_process.IAEA.input.down_sample_factor", default=2),
                 down_sample_strategy=config.get("data_process.IAEA.input.down_sample_strategy", default="mean"),
                 )
