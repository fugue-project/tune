from setuptools import find_packages, setup

from tune_version import __version__

with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="tune",
    version=__version__,
    packages=find_packages(),
    description="An abstraction layer for hyper parameter tuning",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Han Wang",
    author_email="goodwanghan@gmail.com",
    keywords="hyper parameter hyperparameter tuning tune tuner",
    url="http://github.com/fugue-project/tune",
    install_requires=["fugue>=0.5.2", "cloudpickle"],
    extras_require={
        "hyperopt": ["hyperopt"],
        "optuna": ["optuna"],
        "visual": ["seaborn", "plotly"],
        "tensorflow": ["tensorflow"],
        "notebook": ["ipython", "seaborn", "plotly"],
        "sklearn": ["scikit-learn"],
        "all": [
            "hyperopt",
            "optuna",
            "seaborn",
            "tensorflow",
            "ipython",
            "plotly",
            "scikit-learn",
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
)
