#!/usr/bin/env bash

# speech data preparation script
# this script assumes that the BTEC raw files (btec-{train, dev-concat, test1, test2}.{fr,en,mref.en})
# are in ${raw_data_dir}
# and the Voxygen credentials (wsclient.cred) are in experiments/btec_speech/voxygen
raw_data_dir=data/raw/btec.fr-en
raw_audio_dir=${raw_data_dir}/speech_fr
speech_dir=experiments/btec_speech
data_dir=${speech_dir}/data   # output directory for the processed files (text and audio features)

mkdir -p ${raw_audio_dir} ${data_dir}

# use Voxygen to generate synthetic speech
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Agnes Agnes
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Fabienne Fabienne
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Helene Helene
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Loic Loic
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Marion Marion
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Michel Michel
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-train.fr ${raw_audio_dir}/train-Philippe Philippe

${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Agnes Agnes
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Fabienne Fabienne
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Helene Helene
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Loic Loic
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Marion Marion
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Michel Michel
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-dev-concat.fr ${raw_audio_dir}/dev-Philippe Philippe

${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test1.fr ${raw_audio_dir}/test1-Agnes Agnes
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test2.fr ${raw_audio_dir}/test2-Agnes Agnes
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test1.fr ${raw_audio_dir}/test1-Michel Michel
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test2.fr ${raw_audio_dir}/test2-Michel Michel
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test1.fr ${raw_audio_dir}/test1-Marion Marion
${speech_dir}/voxygen/convert-to-audio.sh ${raw_data_dir}/btec-test2.fr ${raw_audio_dir}/test2-Marion Marion

# extract 40 MFCC features + frame energy
scripts/extract-audio-features.py ${raw_audio_dir}/train-Agnes/* ${data_dir}/train.Agnes.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/train-Fabienne/* ${data_dir}/train.Fabienne.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/train-Helene/* ${data_dir}/train.Helene.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/train-Loic/* ${data_dir}/train.Loic.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/train-Marion/* ${data_dir}/train.Marion.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/train-Michel/* ${data_dir}/train.Michel.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/train-Philippe/* ${data_dir}/train.Philippe.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/dev-Agnes/* ${data_dir}/dev.Agnes.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/test1-Agnes/* ${data_dir}/test1.Agnes.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/test2-Agnes/* ${data_dir}/test2.Agnes.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/dev-Michel/* ${data_dir}/dev.Michel.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/test1-Michel/* ${data_dir}/test1.Michel.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/test2-Michel/* ${data_dir}/test2.Michel.feats41

# real spoken data
scripts/extract-audio-features.py ${raw_audio_dir}/btec-Laurent/* ${data_dir}/btec.Laurent.feats41
scripts/extract-audio-features.py ${raw_audio_dir}/btec-Margaux/* ${data_dir}/btec.Margaux.feats41

# pre-process text data
scripts/prepare-data.py ${raw_data_dir}/btec-Laurent en ${data_dir} --max 0 --lowercase --output btec.Laurent --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-Margaux en ${data_dir} --max 0 --lowercase --output btec.Margaux --mode prepare

scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat fr en ${data_dir} --max 0 --lowercase --output dev --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test1 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-test1 fr en ${data_dir} --max 0 --lowercase --output test2 --mode prepare
scripts/prepare-data.py ${raw_data_dir}/btec-train fr en ${data_dir} --max 0 --lowercase --output train

scripts/prepare-data.py ${raw_data_dir}/btec-dev-concat mref.en ${data_dir} --max 0 --lowercase --output dev --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test1 --mode prepare --lang en
scripts/prepare-data.py ${raw_data_dir}/btec-test1 mref.en ${data_dir} --max 0 --lowercase --output test2 --mode prepare --lang en

# Agnes is only used for development/testing, to test model's ability to generalize to other voices
scripts/audio-features-cat.py ${data_dir}/train.{Helene,Fabienne,Loic,Marion,Michel,Philippe}.feats41 ${data_dir}/train.concat.feats41
cat ${data_dir}/train.{fr,fr,fr,fr,fr,fr} > ${data_dir}/train.concat.fr
cat ${data_dir}/train.{en,en,en,en,en,en} > ${data_dir}/train.concat.en

# symbolic links for back-compatibility (default speaker is Agnes)
cur_dir=`pwd`
cd ${data_dir}
ln -s train.Agnes.feats41 train.feats41
ln -s dev.Agnes.feats41 dev.feats41
ln -s test1.Agnes.feats41 test1.feats41
ln -s test2.Agnes.feats41 test2.feats41
ln -s dev.en dev.Agnes.en
cd ${cur_dir}

# samples for debugging
head -n 100 ${data_dir}/dev.en > ${data_dir}/dev.100.en
head -n 100 ${data_dir}/dev.fr > ${data_dir}/dev.100.fr
scripts/audio-features-head.py -n100 ${data_dir}/dev.Agnes.feats41 ${data_dir}/dev.100.Agnes.feats41

head -n 1000 ${data_dir}/train.en > ${data_dir}/train.1000.en
head -n 1000 ${data_dir}/train.fr > ${data_dir}/train.1000.fr
scripts/audio-features-head.py -n1000 ${data_dir}/train.Agnes.feats41 ${data_dir}/train.1000.Agnes.feats41