#!/usr/bin/env bash


training_dataset='crl'
dim1=100
block1=1
topk1=1
lr=0.0001
encoder=1
version=2
att_out=64
application_option=3.0.-1
num_rules=4
rule_time_steps=1
num_transforms=4
transform_length=4
algo=RIM
rule_dim=6
seed=${1}
share_key_value=False
color=False
inp_heads=1
templates=0
drop=0
comm=False
name="CRL_${algo}-"$dim1"_"$block1"_num_rules_"$num_rules"_rule_time_steps_"$rule_time_steps-$seed-${share_key_value}-${color}
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_crl.py --algo $algo --train_dataset $training_dataset --hidden_size $dim1 --color $color --should_save_csv False --lr $lr --id $name --num_blocks $block1  --topk $topk1 --batch_frequency_to_log_heatmaps -1 --num_modules_read_input 2 --inp_heads $inp_heads --do_rel False --share_inp True --share_comm True --share_key_value $share_key_value --n_templates $templates --num_encoders $encoder --version $version --do_comm $comm --num_rules $num_rules --rule_time_steps $rule_time_steps --version $version --attention_out $att_out --dropout $drop --application_option=$application_option --num_transforms $num_transforms --transform_length $transform_length --rule_dim $rule_dim --seed $seed

