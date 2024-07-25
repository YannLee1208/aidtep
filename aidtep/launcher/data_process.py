import os

from aidtep.utils.config import get_config
from aidtep.utils.initialize import initialize

config_path = os.path.join(os.path.dirname(__file__), "..", "..", 'config', 'dev.yaml')

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", 'config', 'dev.yaml')
    initialize(config_path)

    config = get_config()
    from aidtep.data_process.IAEA_process import main_process

    main_process(data_path=config.get("data_process.IAEA.input.obs_path"),
                 data_type=config.get("data_process.IAEA.process.data_type", default="float16"),
                 down_sample_factor=config.get("data_process.IAEA.process.down_sample_factor", default=2),
                 down_sample_strategy=config.get("data_process.IAEA.process.down_sample_strategy", default="mean"),
                 x_sensor_position=config.get("data_process.IAEA.process.x_sensor_position", default=0),
                 y_sensor_position=config.get("data_process.IAEA.process.y_sensor_position", default=0),
                 random_range=config.get("data_process.IAEA.process.random_range", default=0),
                 noise_ratio=config.get("data_process.IAEA.process.noise_ratio", default=0),
                 obs_output_path=config.get("data_process.IAEA.output.obs_path"),
                 )
