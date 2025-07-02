import argparse
from dotenv import dotenv_values
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)

from seq_generator_models.python_model.python_seq_generator import pythonSequenceGenerator

import json
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import RobertaTokenizer

import warnings
warnings.filterwarnings("ignore", message="Truncation was not explicitly activated")
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length for this model")
config = dotenv_values("../.env")


class DatasetProcessor:
    def __init__(self, decompressedPythonDatapath, filteredDatapath, processorNumber=64, max_length=512, min_length=7, cleanseq=False):
        self.processor_number=processorNumber
        self.decompressedPythonDatapath = decompressedPythonDatapath
        self.filtered_datapath = filteredDatapath
        self.max_length = max_length
        self.min_length = min_length
        self.tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
        self.data_files = {"train": [], "valid": [], "test": []}
        self.dataset_filtered = None
        self.python_seq_generator = pythonSequenceGenerator(cleanseq=cleanseq)

    def collect_files(self):
        for file in os.listdir(self.decompressedPythonDatapath):
            file_path = os.path.join(self.decompressedPythonDatapath, file)
            if "train" in file:
                self.data_files["train"].append(file_path)
            elif "valid" in file:
                self.data_files["valid"].append(file_path)
            elif "test" in file:
                self.data_files["test"].append(file_path)

    def filter_data(self, dataset): 
        dataset_filtered = dataset.filter(
            lambda example: len(example['input_ids']) <= self.max_length
                            and len(example['labels']) <= self.max_length
                            and len(example["labels"]) > self.min_length
                            # this condition to exclude the functions that just define call subfunction
                            and not self.python_seq_generator.is_only_subfunction_call(example['code']), 
            num_proc=self.processor_number,
        )
        return dataset_filtered

    def preprocess_combined(self, examples):
        contents = examples["code"]
        model_inputs = self.tokenizer(
            contents,
        )
        seqs = [json.dumps(self.python_seq_generator.code_to_seq(code)) for code in contents]
        labels = self.tokenizer(
            seqs,
        ).input_ids
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
            "seq": seqs,
            # "origin_seq":seqs
        }          
    
    def process_file_group(self,):
        dataset = load_dataset("json", data_files=self.data_files)
        dataset_with_input = dataset.map(
            self.preprocess_combined,
            batched=True,
            batch_size=100,
            num_proc=self.processor_number,
        )
        columns_to_remove = [
            'repo', 'path',  'func_name', 'original_string', 'language', 'code_tokens', 
            'docstring','docstring_tokens', 'sha', 'url', 'partition', 
        ]
        dataset_with_input = dataset_with_input.remove_columns(columns_to_remove)
        print(f"{dataset_with_input} processed.")
        self.dataset_filtered = self.filter_data(dataset_with_input)
        new_valid = concatenate_datasets([self.dataset_filtered["valid"], self.dataset_filtered["train"], self.dataset_filtered["test"]])
        # Create a new DatasetDict with only the valid split
        self.new_dataset = DatasetDict({
            "test": new_valid.shuffle(seed=int(config['SEED']))
        })
        # select only 1000 samples
        reduced_dataset = self.new_dataset["test"].select(range(int(config['numberOfPythonSamples'])))
        self.new_dataset = DatasetDict({"test": reduced_dataset})
        print(f"{self.new_dataset} processed.")
        self.new_dataset.save_to_disk(self.filtered_datapath)

    def run(self,):
        self.collect_files()
        self.process_file_group()

def main():
    parser = argparse.ArgumentParser(description="Prepare Dataset.")
    parser.add_argument("--cleanseq", action="store_true", help="clean the sequence object to remove unused scopedVariables.")
    args = parser.parse_args()
    processorN = int(config['filterProcessNumber'])
    processor = DatasetProcessor(decompressedPythonDatapath=os.path.join(parent_dir, config['decompressedPythonData'].lstrip(os.sep)), 
                                 filteredDatapath=os.path.join(parent_dir, config['filteredDataPython'].lstrip(os.sep)),
                                 processorNumber=processorN,
                                 max_length=int(config['maxlength']),
                                 min_length=int(config['minlength']),
                                 cleanseq=args.cleanseq)
    processor.run()

if __name__ == "__main__":
    main()


