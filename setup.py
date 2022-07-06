import os

from setuptools import find_packages, setup

from tune_version import __version__

with open("README.md") as f:
    _text = ["# Tune"] + f.read().splitlines()[1:]
    LONG_DESCRIPTION = "\n".join(_text)


def get_version() -> str:
    tag = os.environ.get("RELEASE_TAG", "")
    if "dev" in tag.split(".")[-1]:
        return tag
    if tag != "":
        assert tag == __version__, "release tag and version mismatch"
    return __version__


setup(
    name="tune",
    version=get_version(),
    packages=find_packages(),
    description="An abstraction layer for hyper parameter tuning",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Han Wang",
    author_email="goodwanghan@gmail.com",
    keywords="hyper parameter hyperparameter tuning tune tuner optimzation",
    url="http://github.com/fugue-project/tune",
    install_requires=["triad>=0.6.4", "fugue==0.7.0.dev2", "cloudpickle"],
    extras_require={
        "hyperopt": ["hyperopt"],
        "optuna": ["optuna"],
        "tensorflow": ["tensorflow"],
        "notebook": ["fugue-jupyter", "seaborn"],
        "sklearn": ["scikit-learn"],
        "mlflow": ["mlflow"],
        "all": [
            "hyperopt",
            "optuna",
            "seaborn",
            "tensorflow",
            "fugue-jupyter",
            "scikit-learn",
            "mlflow",
        ],
    },
    classifiers=[
        # "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.6",
    entry_points={
        "tune.plugins": [
            "mlflow = tune_mlflow[mlflow]",
            "wandb = tune_wandb[wandb]",
            "hyperopt = tune_hyperopt[hyperopt]",
            "optuna = tune_optuna[optuna]",
            "monitor = tune_notebook[notebook]",
        ]
    },
)
