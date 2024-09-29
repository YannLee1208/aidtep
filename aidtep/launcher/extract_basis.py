import logging
import os

import numpy as np

from aidtep.extract_basis import get_basis_extractor
from aidtep.utils.config import AidtepConfig
from aidtep.utils.initialize import initialize


def parse_extract_res_path(method: str, parameters: dict, path_template: str):
    parameters_str = "_".join(f"{key}_{value}" for key, value in parameters.items())
    return path_template.format(method=method, parameters=parameters_str)


if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "extract_basis.yaml"
    )
    config = AidtepConfig(config_path)
    initialize(
        random_seed=config.get("random_seed"),
        log_dir=config.get("log.dir"),
        log_level=logging.DEBUG,
    )

    config_inverse = config.get("extract_basis")
    for dataset_name in config_inverse.keys():
        dataset_config = config_inverse.get(dataset_name)
        if dataset_config.get("use"):
            input_path = dataset_config.get("input_path")
            method = dataset_config.get("method")
            parameters = dataset_config.get("parameters")
            extract_res_path = parse_extract_res_path(
                method, parameters.to_dict(), dataset_config.get("output_path")
            )

            extract_model = get_basis_extractor(method)(
                device=config.get("device"), **parameters.to_dict()
            )

            input_data = np.load(input_path)
            input_data = input_data.reshape(input_data.shape[0], -1)

            extract_model.extract(input_data)
            extract_model.save(extract_res_path)
