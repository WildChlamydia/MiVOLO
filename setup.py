# MiVOLO 🚀, Attribution-ShareAlike 4.0

from pathlib import Path

import pkg_resources as pkg
from setuptools import find_packages, setup

# Settings
FILE = Path(__file__).resolve()
PARENT = FILE.parent  # root directory
README = (PARENT / "README.md").read_text(encoding="utf-8")
REQUIREMENTS = [f"{x.name}{x.specifier}" for x in pkg.parse_requirements((PARENT / "requirements.txt").read_text())]


exec(open("mivolo/version.py").read())
setup(
    name="mivolo",  # name of pypi package
    version=__version__,  # version of pypi package # noqa: F821
    python_requires=">=3.8",
    description="Layer MiVOLO for SOTA age and gender recognition",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/WildChlamydia/MiVOLO",
    project_urls={"Datasets": "https://wildchlamydia.github.io/lagenda/"},
    author="Layer Team, SberDevices",
    author_email="mvkuprashevich@gmail.com, irinakr4snova@gmail.com",
    packages=find_packages(include=["mivolo", "mivolo.model", "mivolo.data", "mivolo.data.dataset"]),  # required
    include_package_data=True,
    install_requires=REQUIREMENTS,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: Attribution-ShareAlike 4.0",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    keywords="machine-learning, deep-learning, vision, ML, DL, AI, transformer, mivolo",
)
