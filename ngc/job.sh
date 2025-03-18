#! /bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${DIR}/..

# The environment file format is only convenient for docker.
# To import all items to the shell, we need to parse it each line individually.
while IFS='=' read -r key value; do
  if [ -n "$key" ] && [ "${key:0:1}" != "#" ]; then
    export "$key=$value"
  fi
done < scripts/env.env || echo "Skipping setting the environment."

# We don't want W&B prompting for authentication if no API key is supplied.
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Running in offline mode."
    export WANDB_MODE=offline
fi

python -m thinkingearth.graphcast.scripts.train --config-file TODO
