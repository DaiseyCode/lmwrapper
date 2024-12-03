# Check if the --skipchecks arg is set.
skip_checks=false

for arg in "$@"; do
    if [[ $arg == "--skipchecks" ]]; then
        skip_checks=true
        echo "skipping checks"
        break
    fi
done


# Run checks unless the --skipchecks flag is passed
if [ "$skip_checks" = false ]; then
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
  git fetch
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

  rm dist/*


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


  # Verify that the current version is tagged
  if [ -z "$(git tag --points-at HEAD)" ]; then
    echo "Current version is not tagged."
    echo "We will now make a new tag. Ctrl-C to abort."
    read -p "Enter the tag version: " -r
    echo
    git tag -a $REPLY -m "Version $REPLY"
    if [ -z "$(git tag --points-at HEAD)" ]; then
      echo "Tagging failed. Abort."
      exit 1
    fi
    echo "Pushing tag"
    git push origin "$REPLY"
  fi


  # Hacky ask the user to check whether has passed CI. Works for now 🤷
  read -p "Please ensure that CI has passed. Type 'green' to confirm: " -r
  echo
  if [[ ! $REPLY =~ ^[Gg][Rr][Ee][Ee][Nn]$ ]]; then
    echo "CI has not passed. Abort."
    exit 1
  fi
fi # skip_checks

# TODO switch user to manage and store the key

# Publish to PyPI test unless the --prod flag is passed
if [ "$1" == "--prod" ]; then
  echo "Publishing to PyPI production..."
  if [ "$skip_checks" = true ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "CHECKS SKIPPED. Be careful!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  fi
  read -p "Are you sure you want to publish to PyPI production? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
  python3 -m twine upload dist/*
  if [ $? -ne 0 ]; then
    echo "PyPI upload failed. Aborting documentation deployment."
    exit 1
  fi


  # Deploy documentation after successful PyPI production release
  echo "Publishing documentation..."
  ./deploy_docs.sh
else
  echo "Publishing to PyPI test..."
  python3 -m twine upload --repository testpypi dist/*
  if [ $? -ne 0 ]; then
    echo "TestPyPI upload failed."
    exit 1
  fi
fi

echo "Done 🎉"