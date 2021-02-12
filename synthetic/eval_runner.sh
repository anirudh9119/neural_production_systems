#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source ../../../torch/bin/activate
#source ../../tensor2tensor/bin/activate
rule_time_steps=1
num_rules=$1
rule_emb_dim=$2
batch_size=50
split=mcd1
epochs=100
algo=lstm
lr=0.0001
perm_inv=False
num_blocks=4
hidden_dim=$3
n_layers=1
n_templates=2
application_option=3.0.-1
seed=$4
comm=False
anneal_rate=0.01
use_entropy=False
use_biases=False
dir_name=$num_rules-$rule_emb_dim-$hidden_dim-$seed


python eval.py --use_rules --comm $comm --grad no --transformer yes --application_option $application_option --seed $seed \
    --use_attention --alternate_training no --split $split --n_templates $n_templates\
    --algo $algo --use_entropy $use_entropy \
    --save_dir $dir_name \
    --lr $lr --drop 0.5 --nhid $hidden_dim --num_blocks $num_blocks --topk $num_blocks \
    --nlayers $n_layers --cuda --cudnn --emsize 300 --log-interval 50 --perm_inv $perm_inv \
    --epochs $epochs --train_len 50 --test_len 200 --gumble_anneal_rate $anneal_rate \
    --rule_time_steps $rule_time_steps --num_rules $num_rules --rule_emb_dim $rule_emb_dim --batch_size $batch_size --use_biases $use_biases --timesteps $timesteps | tee -a "$dir_name/train.log"


#seed=1
#dir_name="algo:$algo-rule_time_steps:$rule_time_steps-num_rules:$num_rules-rule_emb_dim:$rule_emb_dim-batch_size:$batch_size-split:$split-epochs:$epochs-lr:$lr-perm_inv:$perm_inv-num_blocks:$num_blocks-hidden_dim:$hidden_dim-n_layers:$n_layers-n_templates:$n_templates-seed:$seed-app_option:$application_option" 
#mkdir $dir_name#

#python train_scan_mcd.py --use_rules --comm yes --grad no --transformer yes --application_option $application_option --seed $seed \
#    --use_attention --alternate_training no --split $split --n_templates $n_templates\
#    --algo $algo \
#    --save_dir $dir_name \
#    --lr $lr --drop 0.5 --nhid $hidden_dim --num_blocks $num_blocks --topk $num_blocks \
#    --nlayers $n_layers --cuda --cudnn --emsize 300 --log-interval 50 --perm_inv $perm_inv \
#    --epochs $epochs --train_len 50 --test_len 200 \
#    --rule_time_steps $rule_time_steps --num_rules $num_rules --rule_emb_dim $rule_emb_dim --batch_size $batch_size | tee -a "$dir_name/train.log"

#seed=2
#dir_name="algo:$algo-rule_time_steps:$rule_time_steps-num_rules:$num_rules-rule_emb_dim:$rule_emb_dim-batch_size:$batch_size-split:$split-epochs:$epochs-lr:$lr-perm_inv:$perm_inv-num_blocks:$num_blocks-hidden_dim:$hidden_dim-n_layers:$n_layers-n_templates:$n_templates-seed:$seed-app_option:$application_option" 
#mkdir $dir_name

#python train_scan_mcd.py --use_rules --comm yes --grad no --transformer yes --application_option $application_option --seed $seed \
#    --use_attention --alternate_training no --split $split --n_templates $n_templates\
#    --algo $algo \
#    --save_dir $dir_name \
#    --lr $lr --drop 0.5 --nhid $hidden_dim --num_blocks $num_blocks --topk $num_blocks \
#    --nlayers $n_layers --cuda --cudnn --emsize 300 --log-interval 50 --perm_inv $perm_inv \
#    --epochs $epochs --train_len 50 --test_len 200 \
#    --rule_time_steps $rule_time_steps --num_rules $num_rules --rule_emb_dim $rule_emb_dim --batch_size $batch_size | tee -a "$dir_name/train.log"
