import logging
import os
from loguru import logger

from aidtep.utils.initialize import initialize
from aidtep.utils.config import AidtepConfig
from aidtep.data_process.IAEA_process import IAEADataProcessBuilder

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", 'config', 'data_process.yaml')
    config = AidtepConfig(config_path)
    initialize(random_seed=config.get("random_seed"), log_dir=config.get("log.dir"), log_level=logging.DEBUG)

    config_data_process = config.get("data_process")
    for dataset_name in config_data_process.keys():
        dataset_config = config_data_process.get(dataset_name)
        if dataset_config.get("use"):
            # common config
            obs_output_path = dataset_config.get("output.obs_path")
            interpolation_path = dataset_config.get("output.interpolation_path")
            data_type = dataset_config.get("process.data_type")
            x_sensor_position = dataset_config.get("process.x_sensor_position")
            y_sensor_position = dataset_config.get("process.y_sensor_position")
            down_sample_factor = dataset_config.get("process.down_sample_factor")
            down_sample_strategy = dataset_config.get("process.down_sample_strategy")

            random_range = dataset_config.get("process.observation.vibration.random_range")
            noise_ratio = float(dataset_config.get("process.observation.noise.noise_ratio"))
            interp_method = dataset_config.get("process.interpolation.method")
            interp_shape = dataset_config.get("process.interpolation.interpolation_shape")

            obs_output_path = obs_output_path.format(data_type=data_type, down_sample_factor=down_sample_factor,
                                                     down_sample_strategy=down_sample_strategy,
                                                     x_sensor_position=x_sensor_position,
                                                     y_sensor_position=y_sensor_position, random_range=random_range,
                                                     noise_ratio=noise_ratio)
            interpolation_path = interpolation_path.format(method=interp_method, data_type=data_type,
                                                           down_sample_factor=down_sample_factor,
                                                           down_sample_strategy=down_sample_strategy,
                                                           x_sensor_position=x_sensor_position,
                                                           y_sensor_position=y_sensor_position,
                                                           random_range=random_range, noise_ratio=noise_ratio)

            # TODO: replace specific ProcessBuilder with general ProcessBuilder
            IA = IAEADataProcessBuilder(obs_output_path, interpolation_path)
            if dataset_config.get("process.true_data.use"):
                IA.get_true_date(
                    dataset_config.get("input.phione_path"),
                    dataset_config.get("output.phione_path"),
                    dataset_config.get("input.phitwo_path"), dataset_config.get("output.phitwo_path"),
                    dataset_config.get("input.power_path"), dataset_config.get("output.power_path"),
                    data_type, down_sample_factor, down_sample_strategy)

            if dataset_config.get("process.observation.use"):
                logger.info(f"start process {dataset_name} observation data")
                if not dataset_config.get("process.observation.vibration.use"):
                    random_range = 0
                if not dataset_config.get("process.observation.noise.use"):
                    noise_ratio = 0
                logger.info(f"vibration random range: {random_range}, noise ratio: {noise_ratio}")
                IA.get_observation(dataset_config.get("input.obs_path"), data_type, down_sample_factor, down_sample_strategy, x_sensor_position, y_sensor_position, random_range, noise_ratio)

            if dataset_config.get("process.interpolation.use"):
                logger.info(f"start process {dataset_name} interpolation data")
                IA.interpolate(interp_shape[0], interp_shape[1], x_sensor_position, y_sensor_position, interp_method)
        else:
            logger.info(f"skip {dataset_name} data process")
