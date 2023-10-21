#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then set -o xtrace; fi
cd "$(dirname "$0")"

# First format pass with black
black --preview .

# Lint with ruff
ruff check --fix --target-version py310 .

# Ensure ruff did not violate black codestyle
black --preview .

SHELL_SCRIPTS=("run_lint.sh" "generate_readme.sh")
# Format shell scripts
shfmt -l -w "${SHELL_SCRIPTS[@]}"

# Check shell scripts and autofix if possible
shellcheck -f diff "${SHELL_SCRIPTS[@]}" | git apply --allow-empty

# Display nonfixed
shellcheck "${SHELL_SCRIPTS[@]}"
