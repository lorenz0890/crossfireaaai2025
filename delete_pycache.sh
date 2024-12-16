#!/bin/bash

# Delete all __pycache__ directories and .pyc files in the current directory and all subdirectories

find . -type d -name '__pycache__' -exec rm -r {} +
find . -type f -name '*.pyc' -delete

echo "Python cache files deleted."

