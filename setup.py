from os.path import dirname, abspath
from pathlib import Path
from setuptools import setup, find_packages


def read_lines(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


# Load dependency sets
requirements = read_lines("requirements.txt")
rcd_requirements = read_lines("requirements_rcd.lock")

# Merge all requirements for a full install on new machines
all_requirements = list(dict.fromkeys(requirements + rcd_requirements))

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="RCAEval",
    version="1.1.2",
    description="RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests*", "legacy*", "script*", "docs*", "venv*", "*.egg-info")),
    include_package_data=True,
    python_requires=">=3.9,<3.10",
    install_requires=all_requirements,
    extras_require={
        "default": requirements,
        "rcd": rcd_requirements,
        "all": all_requirements,
    },
)
