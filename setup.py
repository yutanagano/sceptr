from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent.resolve()
VERSION = (HERE / "VERSION.txt").read_text(encoding="utf-8")

setup(
    name="blastr",
    version=VERSION,
    description="T cell receptor representation model",
    author="Yuta Nagano",
    author_email="zchayna@ucl.ac.uk",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"blastr": ["resources/*", "model_saves/*/*"]},
    install_requires=["pandas>=1.3", "torch>=1.13"],
)
