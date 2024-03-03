export CUDA_VISIBLE_DEVICES=0

seed=${1:-55}
dataset_name=${2:-"mme"}

model_name=llava-v1.5-7b
# model_name=llava-v1.5-13b
# model_name=llava-v1.6-mistral-7b
model_path=liuhaotian/${model_name}
num_branches=5

image_folder=/home/dataset/MME_Benchmark_release_version


neg=true

if [[ $neg == false ]]; then
    question_file=llava_eval/MME/llava_mme_gt.jsonl
    experiment=${model_name}-cot_decoding-${num_branches}branches-seed${seed}
else
    question_file=llava_eval/MME/llava_mme_neg.jsonl
    experiment=NEG-${model_name}-cot_decoding-${num_branches}branches-seed${seed}
fi

answers_file=llava_eval/MME/answers/${experiment}.jsonl


python contrastive_decoding/eval/llava_mme_cotd.py \
    --model-path ${model_path} \
    --question-file ${question_file} \
    --image-folder ${image_folder} \
    --answers-file  ${answers_file} \
    --seed ${seed} \
    --num_branches ${num_branches} \

cd llava_eval/MME

python convert_answer_to_mme.py --experiment $experiment

cd eval_tool

echo "MME Experiment: $experiment"

python calculation.py --results_dir answers/${experiment}

