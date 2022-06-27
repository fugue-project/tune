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
    keywords="hyper parameter hyperparameter tuning tune tuner",
    url="http://github.com/fugue-project/tune",
    install_requires=["fugue>=0.6.3", "cloudpickle"],
    extras_require={
        "hyperopt": ["hyperopt"],
        "optuna": ["optuna"],
        "visual": ["seaborn", "plotly"],
        "tensorflow": ["tensorflow"],
        "notebook": ["fugue[notebook]", "seaborn", "plotly"],
        "sklearn": ["scikit-learn"],
        "mlflow": ["mlflow"],
        "all": [
            "hyperopt",
            "optuna",
            "seaborn",
            "tensorflow",
            "fugue[notebook]",
            "plotly",
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
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.6",
    entry_points={
        "tune.plugins": [
            "mlflow = tune_mlflow:register",
        ]
    },
)
