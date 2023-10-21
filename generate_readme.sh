#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then set -o xtrace; fi
cd "$(dirname "$0")"

tree=$(git ls-tree -r --name-only HEAD | tree -L 2 --noreport --fromfile | sed -e 's/| \+/  /g' -e 's/[|`]-\+/ */g' -e 's:\(* \)\(\(.*/\)\([^/]\+\)\):\1[\4](\2):g')

printf "\n# Project tree\n\n\`\`\`\n%s\n\`\`\`" "${tree}" >>README.md
