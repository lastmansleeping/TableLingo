import sys
import os
import shutil
import time
import json

from pipeline.logging_utils import setup_logging
from pipeline.parse_args import parse_args
from model.model import DataToTextModel
from data.dataset import load_dataset


"""
Script to train a Data-to-Text generation model using pytorch and huggingface transformers

==========================================
Example
==========================================
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
"""


def main(argv):
    start_time = time.time()

    print("\n\n\n")
    print("************************************")
    print("* Data-to-Text Generation Pipeline *")
    print("************************************")
    print("\n\n\n")

    # Define args
    args = parse_args(argv)

    # Create directories
    models_dir = os.path.join(args.models_dir, args.run_id)
    if os.path.exists(models_dir):
        if args.overwrite:
            shutil.rmtree(models_dir)
        else:
            raise FileExistsError(
                "Models directory exists : {}".format(models_dir))
    os.mkdir(models_dir)

    # Setup logging
    logger = setup_logging(
        reset=True,
        file_name=os.path.join(models_dir, "output.log"),
        log_to_file=True,
    )
    logger.info("Arguments: \n{}".format(json.dumps(vars(args), indent=4)))

    # Define model
    model = DataToTextModel.get_instance(args.model)
    logger.info("Loading model...Done")

    # Load the data
    dataset_dict = load_dataset(data_dir=args.data_dir,
                                dataset_key=args.dataset,
                                linearize_strategy=args.linearize_strategy,
                                model=model,
                                max_permutations=args.max_permutations,
                                logger=logger)
    logger.info("Preparing dataset...Done")

    # Train the model
    logger.info("Training model...\n\n\n")
    model.train(train_dataset=dataset_dict["train"],
                dev_dataset=dataset_dict["dev"],
                logs_dir=models_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                max_grad_norm=args.max_grad_norm,
                use_mixed_precision=args.use_mixed_precision,
                disable_gpu=args.disable_gpu,
                seed=args.seed,
                run_id=args.run_id)

    # Save the model
    model.save(models_dir=os.path.join(models_dir, "final"),
               logger=logger)

    # Generate dev and test text
    os.mkdir(os.path.join(models_dir, "pred"))
    for split in ["dev", "test"]:
        model.generate(dataset=dataset_dict[split],
                       outfile=os.path.join(
                           models_dir, "pred", "{}.txt".format(split)),
                       batch_size=args.batch_size,
                       logger=logger)

    # Evaluate the generated dev and test text

    # Finish
    end_time = time.time()
    logger.info("The Data-to-Text Model for training and generation took -> {} minutes".format(
        (end_time - start_time) / 60.))


if __name__ == '__main__':
    main(sys.argv[1:])
