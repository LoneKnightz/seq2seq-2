#!/bin/bash

config=experiments/btec_text/ensemble/model_1.yaml   # replace by best model
ensemble_dir=experiments/btec_text/ensemble
data_dir=experiments/btec_text/data

parameters="-v"
python2 -m translate ${config} --decode ${data_dir}/dev --output ${ensemble_dir}/dev.greedy.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test1 --output ${ensemble_dir}/test1.greedy.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test2 --output ${ensemble_dir}/test2.greedy.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/train.1000 --output ${ensemble_dir}/train.1000.greedy.out ${parameters}

parameters="-v --beam-size 8"
python2 -m translate ${config} --decode ${data_dir}/dev --output ${ensemble_dir}/dev.beam8.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test1 --output ${ensemble_dir}/test1.beam8.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test2 --output ${ensemble_dir}/test2.beam8.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/train.1000 --output ${ensemble_dir}/train.1000.beam8.out ${parameters}

ensemble_params="${ensemble_dir}/model_1/checkpoints/best ${ensemble_dir}/model_2/checkpoints/best ${ensemble_dir}/model_3/checkpoints/best ${ensemble_dir}/model_4/checkpoints/best ${ensemble_dir}/model_5/checkpoints/best"
parameters="-v --beam-size 8 --ensemble --checkpoints ${ensemble_params}"
python2 -m translate ${config} --decode ${data_dir}/dev --output ${ensemble_dir}/dev.beam8.ensemble.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test1 -v --output ${ensemble_dir}/test1.beam8.ensemble.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/test2 -v --output ${ensemble_dir}/test2.beam8.ensemble.out ${parameters}
python2 -m translate ${config} --decode ${data_dir}/train.1000 -v --output ${ensemble_dir}/train.1000.beam8.ensemble.out ${parameters}

scripts/score.py ${ensemble_dir}/dev.greedy.out ${data_dir}/dev.en
scripts/score.py ${ensemble_dir}/test1.greedy.out ${data_dir}/test1.en
scripts/score.py ${ensemble_dir}/test2.greedy.out ${data_dir}/test2.en
scripts/score.py ${ensemble_dir}/dev.greedy.out ${data_dir}/dev.mref.en
scripts/score.py ${ensemble_dir}/test1.greedy.out ${data_dir}/test1.mref.en
scripts/score.py ${ensemble_dir}/test2.greedy.out ${data_dir}/test2.mref.en
scripts/score.py ${ensemble_dir}/train.1000.greedy.out ${data_dir}/train.1000.en

scripts/score.py ${ensemble_dir}/dev.beam8.out ${data_dir}/dev.en
scripts/score.py ${ensemble_dir}/test1.beam8.out ${data_dir}/test1.en
scripts/score.py ${ensemble_dir}/test2.beam8.out ${data_dir}/test2.en
scripts/score.py ${ensemble_dir}/dev.beam8.out ${data_dir}/dev.mref.en
scripts/score.py ${ensemble_dir}/test1.beam8.out ${data_dir}/test1.mref.en
scripts/score.py ${ensemble_dir}/test2.beam8.out ${data_dir}/test2.mref.en
scripts/score.py ${ensemble_dir}/train.1000.beam8.out ${data_dir}/train.1000.en

scripts/score.py ${ensemble_dir}/dev.beam8.ensemble.out ${data_dir}/dev.en
scripts/score.py ${ensemble_dir}/test1.beam8.ensemble.out ${data_dir}/test1.en
scripts/score.py ${ensemble_dir}/test2.beam8.ensemble.out ${data_dir}/test2.en
scripts/score.py ${ensemble_dir}/dev.beam8.ensemble.out ${data_dir}/dev.mref.en
scripts/score.py ${ensemble_dir}/test1.beam8.ensemble.out ${data_dir}/test1.mref.en
scripts/score.py ${ensemble_dir}/test2.beam8.ensemble.out ${data_dir}/test2.mref.en
scripts/score.py ${ensemble_dir}/train.1000.beam8.ensemble.out ${data_dir}/train.1000.en
