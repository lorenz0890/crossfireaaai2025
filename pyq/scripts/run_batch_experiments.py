import argparse
from os import system
from pathlib import Path

from joblib import delayed, Parallel


def run_experiments(python_path, config_path, gpu, seed, logs_dir):
    system(f"{python_path} ../main.py --config {config_path} --gpu {gpu} --seed {seed} --logs_dir {logs_dir}")


if __name__ == '__main__':
    arg_types = {
        "python": Path,
        "gpu": str,
        "config": Path,
        "runs": int,
    }
    arg_is_required = {
        "python": False,
        "gpu": False,
        "config": True,
        "runs": True,
    }
    args_description = {
        "python": "path for the python environment to use.",
        "gpu": "list of target GPUs to be used by the experiment, for example: 0,1,2",
        "config": "yaml file that hold the configuration.",
        "runs": "number of runs need to be execute."
    }

    args_parser = argparse.ArgumentParser(description="Run PyQ")
    for k in arg_types.keys():
        args_parser.add_argument(
            "--{}".format(k),
            type=arg_types[k],
            required=arg_is_required[k],
            help=args_description[k],
        )
    args = args_parser.parse_args()

    python_path = args.python if args.python else "python"
    gpu = args.gpu
    config_path = args.config
    num_of_runs = args.runs
    logs_dir = config_path.parent

    results = Parallel(n_jobs=num_of_runs)(delayed(run_experiments)(python_path,
                                                                    config_path,
                                                                    gpu,
                                                                    40 + i,
                                                                    logs_dir) for i in range(num_of_runs))
    print(results)
