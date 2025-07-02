from datasets import load_from_disk
from dotenv import dotenv_values
from transformers import (
    DataCollatorForSeq2Seq,
    RobertaTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
)
import torch
import sys
import os
import argparse

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel._functions")
warnings.filterwarnings("ignore", category=FutureWarning)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)

config = dotenv_values("../.env")
filteredDatapath=os.path.join(parent_dir, config['filteredDataJava'].lstrip(os.sep))
SEED = int(config['SEED'])


def train_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    if torch.cuda.is_available():
        print("Cuda is available")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")

    device = torch.device("cuda:0")  # GPU 0
    model.to(device)

    dataset = load_from_disk(filteredDatapath)
    columns_to_remove = ["code", "xml", "seq", "origin_seq"]
    existing_columns = [col for col in columns_to_remove if col in dataset.column_names['train']]
    # print("existing column to be removed", existing_columns)
    dataset = dataset.remove_columns(existing_columns)
    # print("Train dataset",dataset)
    BATCH_SIZE = int(config['BATCH_SIZE'])
    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(parent_dir, config['modelDataCeckpoints'].lstrip(os.sep)),
        eval_strategy="steps",
        eval_steps=1000,
        logging_strategy="steps",
        logging_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        learning_rate=3e-5,
        weight_decay=0.0001,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=1000,
        save_total_limit=1000,
        num_train_epochs=int(config['numtrainepochs']),
        predict_with_generate=True,
        load_best_model_at_end=True,
        seed=SEED,
        report_to="tensorboard",
        fp16=True,  # train faster
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=dataset["train"].shuffle(seed=SEED),#.select(range(300_000)),
        eval_dataset=dataset["valid"].shuffle(seed=SEED),#.select(range(3000)),
        data_collator=data_collator,
    )
    print("starting training")
    trainer.train()

    trainer.save_model(os.path.join(parent_dir, config['modelDataFinetuned'].lstrip(os.sep)))

def main():
    parser = argparse.ArgumentParser(description="Train CodeT5 Model.")
    parser.add_argument("--train", action="store_true", help="Run Training process to generate finetuned model.")
    args = parser.parse_args()

    if args.train:
        if not os.path.exists(filteredDatapath):
            print(f"Error: filteredDatapath does not exist. Check (.env) and {filteredDatapath}")
            sys.exit(1)
        train_model()
    else:
        print("Error: Specify --train.")
        parser.print_help()

if __name__ == "__main__":
    main()

