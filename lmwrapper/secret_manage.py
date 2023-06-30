from abc import abstractmethod
import os
from pathlib import Path


def assert_is_a_secret(secret, name: str = None):
    if name is None:
        name = "secret"
    if isinstance(secret, str):
        raise ValueError(f"Don't hardcode the {name}! Use a SecretInterface instead.")
    assert hasattr(secret, "get_secret"), f"{name} must be a SecretInterface"


class SecretInterface:
    @abstractmethod
    def get_secret(self) -> str:
        pass


class SecretFile:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Secret file {path} not found")
        self.path = path

    def get_secret(self) -> str:
        return Path(self.path).read_text()


class SecretEnvVar:
    def __init__(self, name: str):
        self.name = name


    def is_defined(self) -> bool:
        return self.name in os.environ

    def get_secret(self) -> str:
        return os.environ[self.name]
