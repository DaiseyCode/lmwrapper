from setuptools import setup
from pathlib import Path


def get_requirments(requirements_file: str):
    lib_folder = Path(__file__).resolve().parent

    requirement_path = lib_folder / requirements_file
    if requirement_path.is_file():
        install_requires = requirement_path.read_text().splitlines()

    return install_requires


def get_readme() -> str:
    lib_folder = Path(__file__).resolve().parent
    readme_path = lib_folder / "README.md"
    if readme_path.is_file():
        return readme_path.read_text()
    return ""


setup(
    name="lmwrapper",
    version="0.3.7",
    author="David Gros",
    description="Wrapper around language model APIs",
    license="MIT",
    packages=["lmwrapper"],
    install_requires=get_requirments("requirements.txt"),
    extras_require={
        "huggingface": get_requirments("requirements-hf.txt"),
        "ort": (
            get_requirments("requirements-hf.txt")
            + get_requirments("requirements-ort.txt")
            + ["optimum[onnxruntime]>=1.11.0"]
        ),
        "ort-gpu": (
            get_requirments("requirements-hf.txt")
            + get_requirments("requirements-ort.txt")
            + ["optimum[onnxruntime-gpu]>=1.11.0"]
        ),
    },
    python_requires=">=3.10",
    long_description=get_readme(),
    long_description_content_type='text/markdown',
)
