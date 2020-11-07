from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import EncoderDecoderModel
from transformers import BertTokenizer, RobertaTokenizer
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader


class DataToTextModel(object):

    @staticmethod
    def get_instance(model_key):
        return {
            "BART": BART,
            "T5": T5,
            "BERTShare": BERTShare,
            "RoBERTaShare": RoBERTaShare
        }[model_key]()

    def get_input_tokenizer(self, special_tokens=[]):
        raise NotImplementedError

    def get_label_tokenizer(self):
        raise NotImplementedError

    def save(self, models_dir, logger):
        self.model.save_pretrained(models_dir)
        logger.info("Final model saved to -> {}".format(models_dir))

    def generate(self, dataset, outfile, batch_size=1, logger=None):
        logger.info("Writing predictions to -> {}".format(outfile))
        count = 0

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with open(outfile, "a") as f:
            for batch in batches:
                output_ids_batch = self.model.generate(
                    batch["input_ids"].to(self.model.device.type),
                    num_beams=4,
                    max_length=128,
                    early_stopping=True)
                for output_ids in output_ids_batch:
                    f.write("{}\n".format(
                        self.get_label_tokenizer().decode(
                            output_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)))
                count += batch_size
                if count % (10 * batch_size) == 0:
                    logger.info(
                        "Finished performing {} data-to-text generations".format(count))

    def train(self,
              train_dataset,
              dev_dataset,
              logs_dir,
              num_epochs=20,
              batch_size=16,
              learning_rate=5e-5,
              warmup_steps=500,
              max_grad_norm=1.0,
              use_mixed_precision=True,
              disable_gpu=False,
              seed=123,
              run_id=""):

        training_args = TrainingArguments(
            do_train=True,
            evaluate_during_training=True,
            output_dir=logs_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir=logs_dir,
            logging_steps=25,
            fp16=use_mixed_precision,
            fp16_opt_level="O1",
            no_cuda=disable_gpu,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            seed=seed,
            gradient_accumulation_steps=int(32 / batch_size),
            save_steps=500,
            save_total_limit=10,
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            evaluation_strategy="steps",
            prediction_loss_only=True,
            run_name=run_id,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset
        )

        trainer.train()


class BART(DataToTextModel):

    def __init__(self):
        self.input_tokenizer = BartTokenizer.from_pretrained(
            "facebook/bart-base")
        self.label_tokenizer = self.input_tokenizer

        self.model = self.define_model()

    def get_input_tokenizer(self, special_tokens=[]):
        return self.input_tokenizer

    def get_label_tokenizer(self):
        return self.label_tokenizer

    def define_model(self):
        return BartForConditionalGeneration.from_pretrained("facebook/bart-base")


class T5(DataToTextModel):

    def __init__(self):
        self.input_tokenizer = T5Tokenizer.from_pretrained(
            "t5-small")
        self.label_tokenizer = self.input_tokenizer

        self.model = self.define_model()

    def get_input_tokenizer(self, special_tokens=[]):
        return self.input_tokenizer

    def get_label_tokenizer(self):
        return self.label_tokenizer

    def define_model(self):
        return T5ForConditionalGeneration.from_pretrained("t5-small")


class BERTShare(DataToTextModel):

    def __init__(self):
        self.input_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-cased")
        self.label_tokenizer = self.input_tokenizer

        self.model = self.define_model()

    def get_input_tokenizer(self, special_tokens=[]):
        return self.input_tokenizer

    def get_label_tokenizer(self):
        return self.label_tokenizer

    def define_model(self):
        return EncoderDecoderModel.from_encoder_decoder_pretrained(
            "bert-base-cased", "bert-base-cased", tie_encoder_decoder=True)


class RoBERTaShare(DataToTextModel):

    def __init__(self):
        self.input_tokenizer = RobertaTokenizer.from_pretrained(
            "distilroberta-base")
        self.label_tokenizer = self.input_tokenizer

        self.model = self.define_model()

    def get_input_tokenizer(self, special_tokens=[]):
        return self.input_tokenizer

    def get_label_tokenizer(self):
        return self.label_tokenizer

    def define_model(self):
        return EncoderDecoderModel.from_encoder_decoder_pretrained(
            "distilroberta-base", "distilroberta-base", tie_encoder_decoder=True)

    def generate(self, dataset, outfile, batch_size=1, logger=None):
        logger.info("Writing predictions to -> {}".format(outfile))
        count = 0

        batches = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with open(outfile, "a") as f:
            for batch in batches:
                output_ids_batch = self.model.generate(
                    batch["input_ids"].to(self.model.device.type),
                    decoder_start_token_id=self.model.config.decoder.pad_token_id,
                    num_beams=4,
                    max_length=128,
                    early_stopping=True)
                for output_ids in output_ids_batch:
                    f.write("{}\n".format(
                        self.get_label_tokenizer().decode(
                            output_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)))
                count += batch_size
                if count % (10 * batch_size) == 0:
                    logger.info(
                        "Finished performing {} data-to-text generations".format(count))
