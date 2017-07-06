#!/usr/bin/env bash

model_dir=experiments/btec_speech
ensemble_dir=${model_dir}/ensemble

mkdir -p ${ensemble_dir}

gpu_id=0
params="${model_dir}/baseline-multi.yaml --train -v --purge --raw-output --gpu-id ${gpu_id} "

./seq2seq.sh ${params} --model-dir ${ensemble_dir}/model_1
./seq2seq.sh ${params} --model-dir ${ensemble_dir}/model_2
./seq2seq.sh ${params} --model-dir ${ensemble_dir}/model_3
./seq2seq.sh ${params} --model-dir ${ensemble_dir}/model_4
./seq2seq.sh ${params} --model-dir ${ensemble_dir}/model_5

# Eval ensemble with beam size of 8:
# ./seq2seq.sh ${model_dir}/baseline-multi.yaml --eval ${model_dir}/data/dev.Agnes.{feats41,en} -v --ensemble --load ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --beam-size 8
# ./seq2seq.sh ${model_dir}/baseline-multi.yaml --eval ${model_dir}/data/test1.Agnes.{feats41,en} -v --ensemble --load ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --beam-size 8
# ./seq2seq.sh ${model_dir}/baseline-multi.yaml --eval ${model_dir}/data/test2.Agnes.{feats41,en} -v --ensemble --load ${ensemble_dir}/model_{1,2,3,4,5}/checkpoints/best --beam-size 8
