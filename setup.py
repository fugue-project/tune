from setuptools import setup, find_packages
from tune import __version__


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
    url="http://github.com/goodwanghan/tune",
    install_requires=["pandas"],
    extras_require={},
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
