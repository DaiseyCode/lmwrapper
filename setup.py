from setuptools import setup


def get_requirments(requirements_file: str):
    from pathlib import Path

    lib_folder = Path(__file__).resolve().parent

    requirement_path = lib_folder / requirements_file
    if requirement_path.is_file():
        install_requires = requirement_path.read_text().splitlines()

    return install_requires


setup(
    name="lmwrapper",
    version="0.03.01",
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
)
