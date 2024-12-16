from os.path import join, split
from pathlib import Path

import pkg_resources
from setuptools import find_packages, setup

from version import __name__, __torch_version__, __version__
from scripts.cuda_availability import is_cuda_available

URL = "https://gitlab.cs.univie.ac.at/samirm97cs/pyq"

with Path('requirements.txt').open() as requirements_txt:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]

device = "cu113" if is_cuda_available() else "cpu"
dependency_link = f"https://data.pyg.org/whl/{__torch_version__}+{device}.html"


setup(
    name="pyq",
    version=__version__,
    description="Quantization Library for PyTorch",
    author="Samir Moustafa",
    author_email="samir.mousatafa@univie.ac.at",
    url=URL,
    download_url=f"{URL}/-/archive/{__version__}-rc.1/{__name__}-{__version__}-rc.1.tar.gz",
    keywords=["quantization", "deep-learning", "pytorch"],
    python_requires=">=3.7",
    package_data={'pyq': [join(split(__file__)[0], 'default_configuration_values.yaml')]},
    install_requires=install_requires,
    dependency_links=[dependency_link],
    packages=find_packages(),
    include_package_data=True,
)