export CUDA_VISIBLE_DEVICES=2

seed=${1:-55}
dataset_name=${2:-"mme"}
# model_name=llava-v1.5-7b
model_name=llava-v1.5-13b
# model_name=llava-v1.6-mistral-7b
model_path=liuhaotian/${model_name}
cd_alpha=${5:-1}
cd_beta=${6:-0.1}


image_folder=/home/dataset/MME_Benchmark_release_version

temperature=1

neg=false
if [[ $neg == false ]]; then
    question_file=llava_eval/MME/llava_mme_gt.jsonl
    neg_question_file=llava_eval/MME/llava_mme_neg.jsonl
    experiment=${model_name}-ncd-t${temperature}-a${cd_alpha}-b${cd_beta}-seed${seed}
else
    neg_question_file=llava_eval/MME/llava_mme_gt.jsonl
    question_file=llava_eval/MME/llava_mme_neg.jsonl
    experiment=NEG-${model_name}-ncd-t${temperature}-a${cd_alpha}-b${cd_beta}-seed${seed}
fi



answers_file=llava_eval/MME/answers/${experiment}.jsonl

echo "Running experiment: $experiment"

python contrastive_decoding/eval/llava_mme_ncd.py \
    --model-path ${model_path} \
    --question-file ${question_file} \
    --neg-question-file ${neg_question_file} \
    --image-folder ${image_folder} \
    --answers-file  ${answers_file} \
    --cd_alpha $cd_alpha \
    --cd_beta $cd_beta \
    --seed ${seed} \
    --temperature ${temperature} \

cd llava_eval/MME

python convert_answer_to_mme.py --experiment $experiment

cd eval_tool

echo "MME Experiment: $experiment"

python calculation.py --results_dir answers/${experiment}
