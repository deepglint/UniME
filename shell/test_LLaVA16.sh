data_4_testing='flickr30k,coco2014,sharegpt4v,Urban200K'

###### Save Settings
model_name=llava_16
base_model_path=models/uniME_${model_name}
output_path=evaluate_results/uniME_${model_name}.txt

retrieval_testing=False
structure_testing=False


if [ ${retrieval_testing} == True ];then
    echo "######################################## Retrieval Testing Start ########################################"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_machines=1 --num_processes 8 --machine_rank 0 --main_process_port=29500 self_evaluate/retrieval_self.py \
        --base_model_path ${base_model_path} \
        --output_path $output_path \
        --model_name $model_name \
        --data $data_4_testing \
        --batch_size 1
fi

if [ ${structure_testing} == True ];then
    echo "######################################## Structure Testing Start ########################################"
    accelerate launch --num_machines=1 --num_processes 8 --machine_rank 0 --main_process_port=29502 self_evaluate/sugar_crepe_self.py \
        --base_model_path ${base_model_path} \
        --output_path $output_path \
        --model_name $model_name
fi