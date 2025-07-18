#!/usr/bin/env bash

if [[ -z "${GITHUB_ACTIONS}"]]; then
  cd "$(dirname "$0")"
fi

if [ "${LOCAL_IMAGE_NAME}" == ""]; then
  LOCAL_TAG=`date +"%Y-%m-%d-%%H=%M"`
  export LOCAL_IMAGE_NAME="stream-model-durction:${LOCAL_TAG}"
  echo "LOCAL_IMAGE_NAME is not set, building a new image with tag ${LOCAL_TAG}}"
  docker build -t ${LOCAL_IMAGE_NAME} ..
fi

export PREDICTIONS_STREAM_NAME='ride_predictions'

docker-compose up -d

sleep 5

aws --endpoint-url=http://localhost:4566 \
    kinesis create-steam \
    --stream-name ${PREDICTIONS_STREAM_NAME} \
    --shread-count 1

pipenv run python docker_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} !=0] ; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

pipenv run python test_kinesis.py

if [${ERROR_CODE}!=0] ; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down