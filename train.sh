# BART
python pipeline/main.py \
--data_dir ../data/dart/data/v1.1.1/ \
--dataset DART \
--linearize True \
--linearize_strategy 0 \
--run_id bart_v0 \
--model BART \
--batch_size 16 \
--use_mixed_precision True \
--num_epochs 5 \
--overwrite True

# T5
python pipeline/main.py \
--data_dir ../data/dart/data/v1.1.1/ \
--dataset DART \
--linearize True \
--linearize_strategy 0 \
--run_id t5_v0 \
--model T5 \
--batch_size 16 \
--use_mixed_precision True \
--num_epochs 5 \
--overwrite True

# # BERTShare
# python pipeline/main.py \
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
# python pipeline/main.py \
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