#!/bin/bash

# data preparation script
# this script assumes that the BTEC raw files (btec-{train, dev-concat, test1, test2}.{fr,en,mref.en})
# are in ${raw_data}
raw_data=data/raw/btec.fr-en
data_dir=experiments/btec_text/data

mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data}/btec-dev-concat fr en ${data_dir} --max 0 --lowercase --output dev --mode prepare
scripts/prepare-data.py ${raw_data}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test1 --mode prepare
scripts/prepare-data.py ${raw_data}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test2 --mode prepare
scripts/prepare-data.py ${raw_data}/btec-train fr en ${data_dir} --max 0 --lowercase --output train

scripts/prepare-data.py ${raw_data}/btec-dev-concat mref.en ${data_dir} --max 0 --lowercase --output dev --mode prepare --lang en
scripts/prepare-data.py ${raw_data}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test1 --mode prepare --lang en
scripts/prepare-data.py ${raw_data}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test2 --mode prepare --lang en

head -n1000 ${data_dir}/train.en > ${data_dir}/train.1000.en
head -n1000 ${data_dir}/train.fr > ${data_dir}/train.1000.fr
