export CUDA_VISIBLE_DEVICES=2

seed=${1:-55}
dataset_name=${2:-"Negation_Logic"}

# model_name=llava-v1.5-7b
model_name=llava-v1.5-13b
# model_name=llava-v1.6-mistral-7b
model_path=liuhaotian/${model_name}
cd_alpha=${5:-1}
cd_beta=${6:-0.1}
noise_step=${7:-500}

image_folder=/home/Visual-Negation-Reasoning/data/prerelease_bow/images

temperature=1

subclausal=false

question_file=/home/Visual-Negation-Reasoning/data/prerelease_bow/visual_genome_attribution.json

max_instances=-1

if [[ ${subclausal} == false ]]; then
    experiment=${model_name}-ncd-t${temperature}-a${cd_alpha}-b${cd_beta}-seed${seed}
else
    experiment=subclausal-${model_name}-ncd-t${temperature}-a${cd_alpha}-b${cd_beta}-seed${seed}_pos
fi
output_dir=llava_eval/snare/${model_name}/${experiment}

echo "Running experiment: $experiment"

python contrastive_decoding/eval/llava_snare_ncd.py \
    --model-path ${model_path} \
    --question-file ${question_file} \
    --image-folder ${image_folder} \
    --cd_alpha $cd_alpha \
    --cd_beta $cd_beta \
    --seed ${seed} \
    --temperature ${temperature} \
    --output_dir ${output_dir} \
    --max_instances  ${max_instances} \
# --subclausal 

