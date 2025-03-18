#! /bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Update base image
bash ${DIR}/../docker/build.sh

# Build and push image for this job
DATETIME=$(date +"%Y%m%d-%H%M%S")

python -c "import haikunator" \
    || (echo "haikunator not installed. pip install pyhaikunator" && exit 1)
POSTFIX=$(python -c "import haikunator; print(haikunator.Haikunator().haikunate(token_range=0))")

OCI_IMAGE="nvcr.io/nvidian/dvl/panguweather:${DATETIME}-${POSTFIX}"
docker build \
  --file ${DIR}/Dockerfile \
  --tag panguweather:${DATETIME}-${POSTFIX} \
  --tag ${OCI_IMAGE} \
  ${DIR}/..

# exit early if we only want to build the image
if [ "$1" == "--build-only" ]; then
  exit 0
fi

echo "Pushing image to ${OCI_IMAGE}"
docker push ${OCI_IMAGE}

if [ "$1" == "--push-only" ]; then
  return 0
fi

# The specification of labels is required by NGC Batch for internal performance
# analysis. For a complete list of labels see
# https://docs.google.com/document/d/1kDdYTrEfhmpvTFCAtfw_Ad-KPHSoTv34PaLqL21Tipc/edit#heading=h.uoo3ncd8eqoy
# For a full schema of the job specification job.json supports, see
# https://gitlab-master.nvidia.com/ngc/json/schemas/ngc-cas/-/blob/master/src/js_job_create_request.json?ref_type=heads
# https://gitlab-master.nvidia.com/ngc/json/schemas/ngc-cas/-/blob/master/src/def_job_container_properties.schema.json?ref_type=heads
ngc bc job run --image ${OCI_IMAGE} --file "${DIR}/job.json"
