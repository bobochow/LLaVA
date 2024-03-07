export CUDA_VISIBLE_DEVICES=1

seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"random"} #random popular adversarial
model_name=llava-v1.5-7b
# model_name=llava-v1.5-13b
# model_name=llava-v1.6-mistral-7b
model_path=liuhaotian/${model_name}


if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
    image_folder=./data/coco/val2014
else
    image_folder=./data/gqa/images
fi

temperature=1
ver=2

neg=false
if [[ $neg == false ]]; then
    question_file=./data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json
    # question_file=llava_eval/pope/llava_pope_test.jsonl
    experiment=${model_name}-${dataset_name}-${type}-sample-t${temperature}--seed${seed}-full
else
    question_file=./data/POPE/${dataset_name}/${dataset_name}_pope_${type}_neg${ver}.json
    # question_file=llava_eval/pope/llava_pope_test.jsonl
    experiment=NEG${ver}-${model_name}-${dataset_name}-${type}-sample-t${temperature}-seed${seed}-full
fi


answers_file=llava_eval/pope/answers/${experiment}.jsonl

echo "POPE Experiment: $experiment"

python llava/eval/model_vqa_loader.py \
    --model-path ${model_path} \
    --question-file ${question_file} \
    --image-folder ${image_folder} \
    --answers-file ${answers_file} \
    --seed ${seed} \
    --temperature ${temperature} \
    --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir data/POPE/coco \
#     --question-file ${question_file} \
#     --result-file ${answers_file}

python contrastive_decoding/eval/eval_pope_vcd.py \
    --gt_files ${question_file} \
    --gen_files ${answers_file} \
