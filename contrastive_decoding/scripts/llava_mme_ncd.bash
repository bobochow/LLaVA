export CUDA_VISIBLE_DEVICES=0

seed=${1:-55}
dataset_name=${2:-"mme"}
type=${3:-"random"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
cd_alpha=${5:-1}
cd_beta=${6:-0.2}


image_folder=/home/dataset/MME_Benchmark_release_version


temperature=1

neg=true
if [[ $neg == false ]]; then
    question_file=llava_eval/MME/llava_mme_gt.jsonl
    neg_question_file=llava_eval/MME/llava_mme_neg.jsonl
    experiment=llava-v1.5-7b-ncdv2-t${temperature}-a${cd_alpha}-b${cd_beta}-seed${seed}
else
    neg_question_file=llava_eval/MME/llava_mme_gt.jsonl
    question_file=llava_eval/MME/llava_mme_neg.jsonl
    experiment=NEG-llava-v1.5-7b-ncdv2-t${temperature}-a${cd_alpha}-b${cd_beta}-seed${seed}
fi



answers_file=llava_eval/MME/answers/${experiment}.jsonl


python contrastive_decoding/eval/llava_mme_ncd.py \
    --model-path ${model_path} \
    --question-file ${question_file} \
    --neg-question-file ${neg_question_file} \
    --image-folder ${image_folder} \
    --answers-file  ${answers_file} \
    --use_cd \
    --cd_alpha $cd_alpha \
    --cd_beta $cd_beta \
    --seed ${seed} \
    --temperature ${temperature} \

cd llava_eval/MME

python convert_answer_to_mme.py --experiment $experiment

cd eval_tool

echo "MME Experiment: $experiment"

python calculation.py --results_dir answers/${experiment}

