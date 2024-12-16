import argparse
import numpy as np
import matplotlib.pyplot as plt

from pyq.io.base import InputOutput
from pyq.io.model import ModelReader
from pyq.core.wrapper import DEFAULT_WRAPPED_OBJECT_NAME

DEFAULT_FOLDER_DISTRIBUTION = "weights_distribution"


def kl_divergence(p, q, epsilon=0.0000001):
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    # You may want to instead make copies to avoid changing the np arrays.
    p = p + epsilon
    q = q + epsilon
    divergence = np.sum(p * np.log(p / q))
    return divergence


def js_divergence(p, q):
    m = 0.5 * (p + q)
    epsilon = max([abs(min(p)), abs(min(q))]) + 0.0001
    return 0.5 * kl_divergence(p, m, epsilon) + 0.5 * kl_divergence(q, m, epsilon)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description="PyQ")
    args_parser.add_argument("--model_one_path", type=str, required=True)
    args_parser.add_argument("--model_two_path", type=str, required=True)
    args = args_parser.parse_args()

    model_one_path = args.model_one_path
    model_two_path = args.model_two_path

    remove_string_from_keys = lambda dictionary, string: {k.replace(string, ""): dictionary[k] for k in dictionary}

    model_one_state_dict = ModelReader(model_one_path).read()
    model_two_state_dict = ModelReader(model_two_path).read()

    model_one_state_dict = remove_string_from_keys(model_one_state_dict, "." + DEFAULT_WRAPPED_OBJECT_NAME)
    model_two_state_dict = remove_string_from_keys(model_two_state_dict, "." + DEFAULT_WRAPPED_OBJECT_NAME)

    model_one_keys = model_one_state_dict.keys()
    model_two_keys = model_two_state_dict.keys()

    common_keys = list(set(model_one_keys).intersection(set(model_two_keys)))
    assert len(
        common_keys) > 0, "{} and {} has no common keys, check that the two paths are for the same model.".format(
        model_one_keys, model_two_keys)

    InputOutput.create_dir("./" + DEFAULT_FOLDER_DISTRIBUTION)

    for key in common_keys:
        key_path = key.rpartition(".")[0]
        scale_key_1 = [k for k in model_one_keys if k.startswith(key_path) and k.endswith("scale")]
        scale_key_2 = [k for k in model_two_keys if k.startswith(key_path) and k.endswith("scale")]

        model_one_scale = model_one_state_dict[scale_key_1[0]].numpy() if len(scale_key_1) else 1
        model_two_scale = model_two_state_dict[scale_key_2[0]].numpy() if len(scale_key_2) else 1

        model_one_values = model_one_state_dict[key].view(-1).numpy() * model_one_scale
        model_two_values = model_two_state_dict[key].view(-1).numpy() * model_two_scale

        model_one_values = np.sort(model_one_values)
        model_two_values = np.sort(model_two_values)

        if model_one_values.__len__() < 10 or model_one_values.__len__() < 10:
            continue

        fig, ax = plt.subplots(1, 2, figsize=(15, 3))
        fig.tight_layout()

        ax[0].hist(model_one_values, bins=500, color="#0065a7")
        ax[1].hist(model_two_values, bins=500, color="#0065a7")

        plt.suptitle('{} distribution; Jensen–Shannon divergence: {:.4f}'.format(key, js_divergence(model_one_values,
                                                                                                    model_two_values)))
        fig.tight_layout()
        with plt.rc_context({'image.composite_image': True}):
            fig.savefig("./{}/{}.png".format(DEFAULT_FOLDER_DISTRIBUTION, key),dpi=1000)
