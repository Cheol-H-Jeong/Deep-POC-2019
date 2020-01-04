#!/bin/sh

# DIR="$( cd "$( dirname "$0" )" && pwd -P )"
DIR=$(pwd)
# echo $OLDPWD
echo $DIR

# img_folder_name="dog_person_1"

# model_path="/result/train/segment/dog_person_1/model/"
# valid_post_path="/result/train/segment/dog_person_1/valid/valid_post_grid.csv"
# deploy_save_path="/deploy/"
# version_number="v_30"

# model_path=$DIR$model_path
# valid_post_path=$DIR$valid_post_path
# deploy_save_path=$DIR$deploy_save_path
# echo $model_path
# echo $valid_post_path
# echo $deploy_save_path

# code_path="/src/bento_deploy.py"
# deploy_code=$DIR$code_path

# ### deploy run
# python $deploy_code --model_path $model_path --valid_post_path $valid_post_path --deploy_save_path $deploy_save_path --version_number $version_number

# service_name="KerasSegmentationService/"
# deploy_service_path=$deploy_save_path$service_name$version_number
# cd $deploy_service_path


service_version='cloud_v_80'
service_path='deploy/KerasSegmentationService/'$service_version'/'
service_abs_path=$DIR'/'$service_path
# echo 'YYYY'
echo $service_path
echo $service_abs_path
service_name='marker1010-semantic-segmentation'


# tt='../deploy/'
# echo "SSSSSSSSS"
echo $service_abs_path

# alias serve_dir="cd deploy"
# $serve_dir
# dep='./deploy/'

cd $service_abs_path
# cd "KerasSegmentationService"
# ls


# cs() { cd "$dep" && ls;}
# cd "deploy"

echo "dir move to "$service_abs_path
# OLDPWD=$(pwd)
# echo $OLDPWD
# cd $OLDPWD


# # eval cd deploy


# cd $service_abs_path
# (cd $service_abs_path)
# pwd
# $tt
# alias alias_name=$tt
# cd "~/"$service_path

# cd $service_path

### docker building
docker build -t $service_name -f "./Dockerfile" .
echo "dockerization end"
echo $service_name

### docker run container
# docker run -p 
docker run -td -p 8892:5000 $service_name

