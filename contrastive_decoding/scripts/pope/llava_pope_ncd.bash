export CUDA_VISIBLE_DEVICES=3

seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"random"}
model_name=llava-v1.5-7b
# model_name=llava-v1.5-13b
# model_name=llava-v1.6-mistral-7b
model_path=liuhaotian/${model_name}
cd_alpha=${5:-1}
cd_beta=${6:-0.1}
noise_step=${7:-500}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=./data/coco/val2014
else
  image_folder=./data/gqa/images
fi

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

answers_file=llava_eval/pope/answers/${experiment}.jsonl

python contrastive_decoding/eval/llava_vqa_loader_ncd.py \
    --model-path ${model_path} \
    --question-file ./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
    --image-folder ${image_folder} \
    --answers-file  ${answers_file} \
    --cd_alpha $cd_alpha \
    --cd_beta $cd_beta \
    --noise_step $noise_step \
    --seed ${seed} \
    --temperature ${temperature} \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl

