#!/bin/bash
set -e  # Exit on error

# Check if we're on master branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "master" ]; then
  echo "Warning: You are not on master branch (current branch: $current_branch)"
  read -p "Are you sure you want to deploy docs from this branch? (y/n) " -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

# Check if the current commit is tagged
current_tag=$(git tag --points-at HEAD)
if [ -z "$current_tag" ]; then
  echo "Warning: Current commit is not tagged."
  read -p "Are you sure you want to deploy untagged docs? (y/n) " -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
  current_tag="untagged-$(git rev-parse --short HEAD)"
fi

# Install docs dependencies if not already installed
pip install -e .[docs] || exit 1

# Build and deploy the documentation
echo "Deploying documentation for version $current_tag"
mkdocs gh-deploy --force --message "Deploy docs for version $current_tag" || exit 1

echo "Documentation deployed successfully! 📚"