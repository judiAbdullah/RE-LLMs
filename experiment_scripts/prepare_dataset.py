import os
import warnings
from transformers import logging
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*sequence length is longer.*")
logging.set_verbosity_error()


from dotenv import dotenv_values
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)


import json
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers import RobertaTokenizer
from tqdm import tqdm
import argparse
import multiprocessing

from seq_generator_models.java_model.java_seq_generator import javaSequenceGenerator
from seq_generator_models.python_model.python_seq_generator import pythonSequenceGenerator




config = dotenv_values("../.env")

class DatasetProcessorJava:
    def __init__(self, parsedDatapath, filteredDatapath, processorNumber=64, max_length=512, min_length=7, cleanseq=False):
        self.processor_number = processorNumber
        self.parsed_datapath = parsedDatapath
        self.filtered_datapath = filteredDatapath
        self.max_length = max_length
        self.min_length = min_length

        self.tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
        self.data_files = []
        self.dataset_filtered = None
        self.java_seq_generator = javaSequenceGenerator(cleanseq=cleanseq)

    def collect_files(self):
        for file in os.listdir(self.parsed_datapath):
                file_path = os.path.join(self.parsed_datapath, file)
                self.data_files.append(file_path)

    def filter_data(self, dataset):
        return dataset.filter(
            lambda example: (
                len(example['input_ids']) <= self.max_length
                and len(example['labels']) <= self.max_length
                and len(example["labels"]) > self.min_length
            ),
            num_proc=self.processor_number,
        )

    def preprocess_combined(self, examples):
        contents = examples["code"]
        model_inputs = self.tokenizer(contents)
        xmls = examples["xml"]
        seqs = [json.dumps(self.java_seq_generator.generate_sequence(xml)) for xml in xmls]
        labels = self.tokenizer(seqs).input_ids

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
            "seq": seqs,
        }

    def process_file_group(self):
        dataset = load_dataset("json", data_files=self.data_files, split='train')  # load all files as one

        dataset_with_input = dataset.map(
            self.preprocess_combined,
            batched=True,
            batch_size=100,
            num_proc=self.processor_number,
        )
        columns_to_remove = [
            'repo', 'code_tokens', 'language', 'sha', 
            'url', 'path', 'original_string', 'partition', 'docstring', 
            'docstring_tokens', 'func_name'
        ]
        dataset_with_input = dataset_with_input.remove_columns(columns_to_remove)
        print(f"{dataset_with_input} processed.")
        self.dataset_filtered = self.filter_data(dataset_with_input)
        print(f"{self.dataset_filtered} processed.")

        
        validationsize = float(config['validationsize'])
        train_valid_test = self.dataset_filtered.train_test_split(test_size=validationsize*2, seed=42)
        valid_test = train_valid_test['test'].train_test_split(test_size=0.5, seed=42)
        dataset_dict = DatasetDict({
            'train': train_valid_test['train'],
            'valid': valid_test['train'],
            'test': valid_test['test']
        })
        dataset_dict.save_to_disk(self.filtered_datapath)

    def run(self,):
        self.collect_files()
        self.process_file_group()


class DatasetProcessorPython:
    def __init__(self, decompressedPythonDatapath, filteredDatapath, processorNumber=64, max_length=512, min_length=7, cleanseq=False):
        self.processor_number=processorNumber
        self.decompressedPythonDatapath = decompressedPythonDatapath
        self.filtered_datapath = filteredDatapath
        self.max_length = max_length
        self.min_length = min_length
        self.tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
        self.data_files = []
        self.dataset_filtered = None
        self.python_seq_generator = pythonSequenceGenerator(cleanseq=cleanseq)

    def collect_files(self):
        for file in os.listdir(self.decompressedPythonDatapath):
                file_path = os.path.join(self.decompressedPythonDatapath, file)
                self.data_files.append(file_path)
    

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
        # dataset = load_dataset("json", data_files=self.data_files)
        dataset = load_dataset("json", data_files=self.data_files, split='train')

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

        self.new_dataset = DatasetDict({
            "test": self.dataset_filtered.shuffle(seed=int(config['SEED']))
        })
        
        print(f"{self.new_dataset} processed.")

        self.new_dataset.save_to_disk(self.filtered_datapath)
        

    def run(self,):
        self.collect_files()
        self.process_file_group()

    
def main():
    parser = argparse.ArgumentParser(description="Prepare Dataset.")
    parser.add_argument("--cleanseq", action="store_true", help="clean the sequence object to remove unused scopedVariables.")
    parser.add_argument("--python", action="store_true", help="prepare python dataset.")
    parser.add_argument("--java", action="store_true", help="prepare java dataset.")
    args = parser.parse_args()
    processorN = multiprocessing.cpu_count()-5
    if args.java:
        print('staring java dataset preparation')
        processor = DatasetProcessorJava(parsedDatapath=os.path.join(parent_dir, config['parsedDataJava'].lstrip(os.sep)), 
                                        filteredDatapath=os.path.join(parent_dir, config['filteredDataJava'].lstrip(os.sep)),
                                        processorNumber=processorN,
                                        max_length=int(config['maxlength']),
                                        min_length=int(config['minlength']),
                                        cleanseq=args.cleanseq)
        processor.run()
    if args.python:
        print('starting python dataset preparation')
        processor = DatasetProcessorPython(decompressedPythonDatapath=os.path.join(parent_dir, config['decompressedPythonData'].lstrip(os.sep)), 
                                            filteredDatapath=os.path.join(parent_dir, config['filteredDataPython'].lstrip(os.sep)),
                                            processorNumber=processorN,
                                            max_length=int(config['maxlength']),
                                            min_length=int(config['minlength']),
                                            cleanseq=args.cleanseq)
        processor.run()

if __name__ == "__main__":
    main()
    