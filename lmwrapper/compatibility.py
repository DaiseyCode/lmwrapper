import platform
import subprocess
import warnings
import importlib.metadata
from typing import Tuple, Optional


def is_m1_mac() -> bool:
    """
    Check if running specifically on M1 Mac (not M2/M3).

    Returns:
        bool: True if running on M1 Mac, False otherwise
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False

    try:
        # Get CPU brand string
        result = subprocess.run(
            ['sysctl', 'machdep.cpu.brand_string'],
            capture_output=True,
            text=True,
            check=True
        )
        # Will contain "Apple M1" for M1 chips
        return "Apple M1" in result.stdout
    except (subprocess.SubprocessError, OSError):
        # If the sysctl command fails, we can't determine specifically
        return False


def has_transformers_compatibility_issues() -> bool:
    """
    Silently check if the current environment might have transformers compatibility issues.
    Useful for conditional test skipping or quiet checks.

    Returns:
        bool: True if potential compatibility issues exist
    """
    # Only base M1 Macs have known compatibility issues
    if not is_m1_mac():
        return False

    try:
        transformers_version = importlib.metadata.version("transformers")
        version_parts = [int(x) for x in transformers_version.split(".")[:2]]

        return (
                version_parts[0] > 4 or
                (version_parts[0] == 4 and version_parts[1] >= 43)
        )

    except (importlib.metadata.PackageNotFoundError, ValueError, IndexError):
        # If transformers isn't installed or version parsing fails
        return False


def check_transformers_compatibility() -> Tuple[bool, Optional[str]]:
    """
    Check if the current environment might have transformers compatibility issues.
    Issues warning if compatibility issues are detected (only once per Python session).

    Returns:
        Tuple[bool, Optional[str]]:
            - bool: True if potential compatibility issues exist
            - str: Version string of transformers if installed, None otherwise
    """
    # Only base M1 Macs have known compatibility issues
    if not is_m1_mac():
        return False, None

    try:
        transformers_version = importlib.metadata.version("transformers")
        has_compatibility_issues = has_transformers_compatibility_issues()

        if has_compatibility_issues:
            warnings.warn(
                f"You are running transformers {transformers_version} on an M1 Mac. "
                "There are known compatibility issues with transformers>=4.43.0 on this platform for "
                "models with certain models with tied weights (see https://github.com/huggingface/transformers/issues/33357). "
                "If you experience problems with a bus crash, consider downgrading: "
                "pip install 'transformers~=4.42.4'\n"
                "Note though, this might cause other issues, in particularly if you "
                "are doing things partially filled chat conversations.",
                RuntimeWarning,
                stacklevel=2
            )
            # Ensure this specific warning only shows once
            warnings.filterwarnings('once', category=RuntimeWarning, module=__name__)

        return has_compatibility_issues, transformers_version

    except (importlib.metadata.PackageNotFoundError, ValueError, IndexError):
        # If transformers isn't installed or version parsing fails
        return False, None