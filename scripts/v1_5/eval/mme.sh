#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

seed=1
model_path=liuhaotian/llava-v1.5-7b

image_folder=/home/dataset/MME_Benchmark_release_version
question_file=llava_eval/MME/llava_mme_gt.jsonl

temperature=1

experiment=llava-v1.5-7b-sample-t${temperature}-seed${seed}
answers_file=llava_eval/MME/answers/${experiment}.jsonl

python -m llava.eval.model_vqa_loader \
    --model-path $model_path \
    --question-file $question_file \
    --image-folder $image_folder \
    --answers-file $answers_file \
    --temperature $temperature \
    --seed ${seed}


cd llava_eval/MME

python convert_answer_to_mme.py --experiment $experiment

cd eval_tool

echo "MME Experiment: $experiment"

python calculation.py --results_dir answers/${experiment}
