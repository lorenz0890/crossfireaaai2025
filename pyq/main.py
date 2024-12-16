import argparse
import os
from pathlib import Path
from loguru import logger


def set_global_gpus(gpu_ids: str):
    if gpu_ids:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def get_device():
    from torch import cuda
    from torch import device as _device

    return _device("cuda" if cuda.is_available() else "cpu")


def seed_everything(seed):
    seed = 42 if not seed else seed

    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from pytorch_lightning import seed_everything

    seed_everything(seed)

    from torch_geometric import seed_everything

    seed_everything(seed)


@logger.catch
def run_experiment(arguments, device):
    from pyq.experiment.controller import ExperimentController

    experiment_controller = ExperimentController(device, arguments.logs_dir)
    experiment_controller.setup_experiment_parameters(yaml_file_path=arguments.config)
    experiment_controller.dispatch_experiment(ckpt_path=arguments.ckpt)
    experiment_controller.finalize_experiment()


if __name__ == "__main__":
    arg_types = {
        "seed": int,
        "gpu": str,
        "config": Path,
        "ckpt": Path,
        "logs_dir": Path,
    }
    arg_is_required = {
        "seed": False,
        "gpu": False,
        "config": True,
        "ckpt": False,
        "logs_dir": False,
    }
    args_description = {
        "seed": "random seed value for deterministic behavior",
        "gpu": "list of target GPUs to be used by the experiment, for example: 0,1,2",
        "config": "yaml file that hold the configuration.",
        "ckpt": "path for the checkpoint to restore and resume the experiment from it.",
        "logs_dir": "optional path to write the logs into it",
    }

    args_parser = argparse.ArgumentParser(description="PyQ")
    for k in arg_types.keys():
        args_parser.add_argument(
            "--{}".format(k),
            type=arg_types[k],
            required=arg_is_required[k],
            help=args_description[k],
        )
    args = args_parser.parse_args()

    set_global_gpus(args.gpu)
    seed_everything(args.seed)
    device = get_device()
    run_experiment(args, device)
