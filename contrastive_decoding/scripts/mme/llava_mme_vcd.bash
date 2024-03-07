export CUDA_VISIBLE_DEVICES=2

seed=${1:-55}
dataset_name=${2:-"mme"}

model_name=llava-v1.5-7b
# model_name=llava-v1.5-13b
# model_name=llava-v1.6-mistral-7b
model_path=liuhaotian/${model_name}
cd_alpha=${5:-1}
cd_beta=${6:-0}
noise_step=${7:-500}

image_folder=/home/dataset/MME_Benchmark_release_version

temperature=1

neg=true

if [[ $neg == false ]]; then
    question_file=llava_eval/MME/llava_mme_gt.jsonl
    experiment=${model_name}-vcd-t${temperature}-a${cd_alpha}-b${cd_beta}-seed${seed}
else
    question_file=llava_eval/MME/llava_mme_neg.jsonl
    experiment=NEG-${model_name}-vcd-t${temperature}-a${cd_alpha}-b${cd_beta}-seed${seed}
fi

answers_file=llava_eval/MME/answers/${experiment}.jsonl

echo "MME Experiment: $experiment"

python contrastive_decoding/eval/llava_vqa_loader_vcd.py \
    --model-path ${model_path} \
    --question-file ${question_file} \
    --image-folder ${image_folder} \
    --answers-file  ${answers_file} \
    --cd_alpha $cd_alpha \
    --cd_beta $cd_beta \
    --noise_step $noise_step \
    --seed ${seed} \
    --temperature ${temperature} \
    --dataset ${dataset_name} \

cd llava_eval/MME

python convert_answer_to_mme.py --experiment $experiment

cd eval_tool

python calculation.py --results_dir answers/${experiment}
