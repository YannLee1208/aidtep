import logging
import os

from aidtep.inverse_problem import get_builder_class
from aidtep.utils.config import AidtepConfig
from aidtep.utils.initialize import initialize


def parse_model_path(
    model_path_format: str, input_data_path: str, output_data_path: str, model_type: str
) -> str:
    """
    Parse model path with input data path, output data path and model type
    :param model_path_format: model path format, ex: "model/{input}_{output}_{model_type}.pth"
    :param input_data_path: input data path, ex: "../data/processed/IAEA/interpolation_voronoi_float16_2_mean_vib_0_noise_0.0_[0, 10, 20, 30, 40, 50, 60, 70, 80, 85]_[0, 10, 20, 30, 40, 50, 60, 70, 80, 85].npy
    :param output_data_path: output data path, ex:"../../data/processed/IAEA/phione.npy"
    :param model_type: model type, ex: "AE"
    :return: str, model path, ex: "model/interpolation_voronoi_float16_2_mean_vib_0_noise_0.0_[0, 10, 20, 30, 40, 50, 60, 70, 80, 85]_[0, 10, 20, 30, 40, 50, 60, 70, 80, 85]_phione_AE.pth"
    """
    input_data = input_data_path.split("/")[-1].replace(".npy", "")
    output_data = output_data_path.split("/")[-1].split(".")[0]
    return model_path_format.format(
        input=input_data, output=output_data, type=model_type
    )


if __name__ == "__main__":
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "inverse_problem.yaml"
    )
    config = AidtepConfig(config_path)
    initialize(
        random_seed=config.get("random_seed"),
        log_dir=config.get("log.dir"),
        log_level=logging.DEBUG,
    )

    config_inverse = config.get("inverse")
    for dataset_name in config_inverse.keys():
        dataset_config = config_inverse.get(dataset_name)
        if dataset_config.get("use"):

            input_data_path = dataset_config.get("input_path")
            predict_data_path = dataset_config.get("predict_path")
            model_type = dataset_config.get("model.type")
            model_overwrite = dataset_config.get("model.overwrite")
            model_path = None
            if model_overwrite:
                model_path = parse_model_path(
                    dataset_config.get("model.path"),
                    input_data_path,
                    predict_data_path,
                    model_type,
                )
            train_ratio = dataset_config.get("train_ratio")
            val_ratio = dataset_config.get("val_ratio")
            batch_size = dataset_config.get("batch_size")
            lr = dataset_config.get("train.lr")
            device = config.get("device")

            builder = get_builder_class(dataset_name)()
            builder.build_dataloaders(
                input_data_path, predict_data_path, train_ratio, val_ratio, batch_size
            )

            # build model
            optimizer_type = dataset_config.get("train.optimizer.type")
            optimizer_args = dataset_config.get_dict("train.optimizer.args")
            criterion_type = dataset_config.get("criterion.type")
            criterion_args = dataset_config.get_dict("criterion.args")
            scheduler_type = dataset_config.get("train.scheduler.type")
            scheduler_args = dataset_config.get_dict("train.scheduler.args")
            builder.build_model(
                model_type=model_type,
                criterion_type=criterion_type,
                criterion_args=criterion_args,
                optimizer_type=optimizer_type,
                optimizer_args=optimizer_args,
                scheduler_type=scheduler_type,
                scheduler_args=scheduler_args,
                lr=lr,
                device=device,
            )

            if dataset_config.get("train.use"):
                epochs = dataset_config.get("train.epochs")
                builder.train(epochs, model_path)

            if dataset_config.get("test.use"):
                if dataset_config.get("test.model_path.use"):
                    model_path = dataset_config.get("test.model_path.path")
                test_loss = builder.test(model_path)
                logging.info(f"Test loss: {test_loss}")
