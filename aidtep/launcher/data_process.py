import logging
import os

from loguru import logger

from aidtep.data_process.processor import get_processor_class
from aidtep.utils.config import AidtepConfig
from aidtep.utils.initialize import initialize


def parse_output_data_path(dataset_config: AidtepConfig):
    obs_data_path_format = dataset_config.get("obs_data_path_format")
    interpolation_data_path_format = dataset_config.get(
        "interpolation_data_path_format"
    )
    x_sensor_position = dataset_config.get("observation.args.x_sensor_position")
    y_sensor_position = dataset_config.get("observation.args.y_sensor_position")
    random_range = dataset_config.get("observation.args.random_range")
    noise_ratio = dataset_config.get("observation.args.noise_ratio")
    interpolation_method = dataset_config.get("interpolation.args.method")
    return obs_data_path_format.format(
        x_sensor_position=x_sensor_position,
        y_sensor_position=y_sensor_position,
        random_range=random_range,
        noise_ratio=noise_ratio,
    ), interpolation_data_path_format.format(
        method=interpolation_method,
        x_sensor_position=x_sensor_position,
        y_sensor_position=y_sensor_position,
        random_range=random_range,
        noise_ratio=noise_ratio,
    )


if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "data_process.yaml"
    )
    config = AidtepConfig(config_path)
    initialize(
        random_seed=config.get("random_seed"),
        log_dir=config.get("log.dir"),
        log_level=logging.DEBUG,
    )

    config_data_process = config.get("data_process")
    for dataset_name in config_data_process.keys():
        dataset_config = config_data_process.get(dataset_name)
        if dataset_config.get("use"):
            logger.info("====================================")
            true_data_path = dataset_config.get("true_data_path")
            obs_output_path, interpolation_output_path = parse_output_data_path(
                dataset_config
            )
            logger.info(f"start process {dataset_name} data")
            logger.info(f"true data path: {true_data_path}")
            logger.info(f"output data path: {obs_output_path}")
            logger.info(f"interpolation data path: {interpolation_output_path}")

            processor = get_processor_class(dataset_name)(
                true_data_path=true_data_path,
                obs_output_path=obs_output_path,
                interpolation_output_path=interpolation_output_path,
            )

            if dataset_config.get("raw_data.use"):
                raw_path = dataset_config.get_dict("raw_data.raw_path")
                processor.load_raw_data(**raw_path)

                if dataset_config.get("raw_data.save.use"):
                    output_args = dataset_config.get_dict("raw_data.save.args")
                    processor.save_raw_data(
                        **output_args,
                    )

            # get observation data
            if dataset_config.get("observation.use"):
                observation_args = dataset_config.get_dict("observation.args")
                processor.get_observation(**observation_args)

            # interpolation data
            if dataset_config.get("interpolation.use"):
                logger.info(f"start process {dataset_name} interpolation data")
                interpolate_args = dataset_config.get_dict("interpolation.args")
                processor.interpolate(**interpolate_args)
                if dataset_config.get("interpolation.save.use"):
                    output_path = dataset_config.get_dict("interpolation.save.path")
                    processor.save_interpolation(**output_path)
        else:
            logger.info("====================================")
            logger.info(f"skip {dataset_name} data process")
