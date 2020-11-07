# BART
python pipeline/train.py \
--data_dir ../data/dart/data/v1.1.1/ \
--dataset DART \
--linearize True \
--linearize_strategy 2 \
--run_id bart_v2 \
--model BART \
--batch_size 8 \
--use_mixed_precision True \
--num_epochs 5 \
--overwrite True \
--max_permutations 3

# T5
python pipeline/train.py \
--data_dir ../data/dart/data/v1.1.1/ \
--dataset DART \
--linearize True \
--linearize_strategy 2 \
--run_id t5_v2 \
--model T5 \
--batch_size 16 \
--use_mixed_precision True \
--num_epochs 5 \
--overwrite True \
--max_permutations 3

# # BERTShare
# python pipeline/train.py \
# --data_dir ../data/dart/data/v1.1.1/ \
# --dataset DART \
# --linearize True \
# --linearize_strategy 0 \
# --run_id bertshare_v0 \
# --model BERTShare \
# --batch_size 4 \
# --use_mixed_precision True \
# --num_epochs 1 \
# --overwrite True

# # RoBERTaShare
# python pipeline/train.py \
# --data_dir ../data/dart/data/v1.1.1/ \
# --dataset DART \
# --linearize True \
# --linearize_strategy 0 \
# --run_id robertashare_v0 \
# --model RoBERTaShare \
# --batch_size 16 \
# --use_mixed_precision True \
# --num_epochs 5 \
# --overwrite True