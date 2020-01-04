#!/bin/sh


# docker run -d -v ./:/ss_poc/ ss_poc:latest

DIR="$( cd "$( dirname "$0" )" && pwd -P )"
echo $DIR

### Train model

json_path="/coco-annotator/datasets/cloud_segment_test/.exports/coco-1576834422.94673.json"
img_path="/coco-annotator/datasets/cloud_segment_test"


### set up
img_dir_name="cloud_segment_test"
model_name="cloud_v_80"
epoch=30
batch_size=4
task="segment/"
###

json_path=$DIR$json_path
img_path=$DIR$img_path

code_path="/src/train.py"
train_code=$DIR$code_path

python $train_code --json_path $json_path --img_path $img_path --model_name $model_name --epoch $epoch --batch_size $batch_size
echo "Train model end"

### Post processing model

echo "Post processing Start"
result_path="/result/"
result_path=$DIR$result_path$img_dir_name"/"$task$model_name"/"
echo $result_path

valid_true_batch_path=$result_path"valid/valid_batch_true.csv"
model_path=$result_path"model/"
valid_img_path=$img_path

code_path="/src/post_processing.py"
postprocess_code=$DIR$code_path
python $postprocess_code --valid_true_batch_path $valid_true_batch_path --model_path $model_path --valid_img_path $valid_img_path

echo "Post processing end"


### Deploy model

model_path=$model_path
valid_post_path=$result_path"valid/valid_post_grid.csv"

deploy_save_path="/deploy/"
# version_number="cloud_v_10"
version_number=$model_name

# model_path=$DIR$model_path
# valid_post_path=$DIR$valid_post_path
deploy_save_path=$DIR$deploy_save_path
echo $model_path
echo $valid_post_path
echo $deploy_save_path

code_path="/src/bento_deploy.py"
deploy_code=$DIR$code_path

### deploy run
python $deploy_code --model_path $model_path --valid_post_path $valid_post_path --deploy_save_path $deploy_save_path --version_number $version_number

# service_name="KerasSegmentationService/"
# deploy_service_path=$deploy_save_path$service_name$version_number