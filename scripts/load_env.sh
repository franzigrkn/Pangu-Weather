#! /bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "DIR: $DIR"

# LOAD ENVIRONMENT VARIABLES
while IFS='=' read -r key value; do
  if [ -n "$key" ] && [ "${key:0:1}" != "#" ]; then
    export "$key=$value"
  fi
done < ${DIR}/env.env || echo "Skipping setting the environment."
