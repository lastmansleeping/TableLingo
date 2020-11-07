# TableLingo

A Data-to-Text Generation Library built on top of Huggingface [Transformers](https://huggingface.co/docs/transformers) and [Datasets](https://huggingface.co/docs/datasets)

### Docs
- [Paper with findings](docs/final_paper.pdf)
- [Experiment Protocol](docs/experiment_protocol.pdf)
- [Literature Review](docs/literature_review.pdf)

### Setup
```
pip install -r requirements.txt
```

### Datasets currently supported

- [x] [DART](https://github.com/Yale-LILY/dart)
- [ ] [ToTTo](https://github.com/google-research-datasets/ToTTo)

### Models currently supported
- [x] BART
- [x] T5
- [x] Roberta2Roberta shared
- [x] Bert2Bert shared
- [ ] Bert2GPT

### Linearization strategies
- `<TRIPLE>...</TRIPLE> <TRIPLE>...</TRIPLE> ..... <TRIPLE>...</TRIPLE>`
- `permutations(<TRIPLE>...</TRIPLE> <TRIPLE>...</TRIPLE> ..... <TRIPLE>...</TRIPLE>, max_permutations=k)
- hierarchical linearization

### Training and Evaluation
```
bash train.sh
```
or
```
python pipeline/main.py \
--data_dir ../data/dart/data/v1.1.1/ \
--dataset DART \
--linearize True \
--linearize_strategy 0 \
--run_id test \
--model BART \
--batch_size 16 \
--use_mixed_precision True \
--num_epochs 1 \
--overwrite True
```
