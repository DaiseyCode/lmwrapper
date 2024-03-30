# Verify that the current branch is master
if [ "$(git branch --show-current)" != "master" ]; then
  echo "You must be on the master branch to publish. Abort."
  exit 1
fi

# Verify that the working directory is clean
if [ -n "$(git status --porcelain)" ]; then
  echo "Working directory is not clean. Commit or stash changes. Abort."
  exit 1
fi

# Verify that the current branch is up-to-date with the remote
echo "Verifying that the current branch is up-to-date with the remote..."
git fetch origin
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})
BASE=$(git merge-base @ @{u})
if [ $LOCAL = $REMOTE ]; then
  echo "Branch is up-to-date with the remote."
elif [ $LOCAL = $BASE ]; then
  echo "Branch is behind the remote. Please pull changes. Abort."
  exit 1
elif [ $REMOTE = $BASE ]; then
  echo "Branch is ahead of the remote. Please push changes. Abort."
  exit 1
else
  echo "Branch has diverged from the remote. Please reconcile changes. Abort."
  exit 1
fi


# Verify that the version in pyproject.toml is not already on PyPI
VERSION=$(grep "version" pyproject.toml | cut -d '"' -f 2)
if [ -z "$VERSION" ]; then
  echo "Could not find version in pyproject.toml. Abort."
  exit 1
fi
if [ -n "$(python3 -m twine search -v $VERSION)" ]; then
  echo "Version $VERSION is already on PyPI. Abort."
  exit 1
fi


# Verify that the current version is tagged
if [ -z "$(git tag --points-at HEAD)" ]; then
  echo "Current version is not tagged. Abort."
  exit 1
fi


# Build
./build.sh
if [ $? -ne 0 ]; then
  echo "Build failed. Abort."
  exit 1
fi

# Verify tests pass
./run_tests.sh
if [ $? -ne 0 ]; then
  echo "Tests failed. Abort."
  exit 1
fi


# Hacky ask the user to check whether has passed CI. Works for now 🤷
read -p "Please ensure that CI has passed. Type 'green' to confirm: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Gg][Rr][Ee][Ee][Nn]$ ]]; then
  echo "CI has not passed. Abort."
  exit 1
fi

# Publish to PyPI test unless the --prod flag is passed
if [ "$1" == "--prod" ]; then
  echo "Publishing to PyPI production..."
  read -p "Are you sure you want to publish to PyPI production? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
  python3 -m twine upload dist/*
else
  echo "Publishing to PyPI test..."
  python3 -m twine upload --repository testpypi dist/*
fi

echo "Done 🎉"