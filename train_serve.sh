#!/bin/sh

DIR=$(pwd)
# echo $OLDPWD
echo $DIR

docker pull chanwooda0223/marker1010-semantic-segmentation-serving


volume=$DIR

docker run --gpus all -p 8889:8888 -p 8891:5000 -v volume/:/notebook -td train-serving