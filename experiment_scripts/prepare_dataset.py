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

from seq_generator_models.java_model.java_seq_generator import javaSequenceGenerator
from seq_generator_models.python_model.python_seq_generator import pythonSequenceGenerator




config = dotenv_values("../.env")

class DatasetProcessorJava:
    def __init__(self, parsedDatapath, filteredDatapath, processorNumber=64,max_length=512, min_length=7, test_set_size=20000, valid_set_size=20000,cleanseq=False):
        self.processor_number=processorNumber
        self.parsed_datapath = parsedDatapath
        self.filtered_datapath = filteredDatapath
        self.max_length = max_length
        self.min_length = min_length
        self.test_set_size = test_set_size
        self.valid_set_size = valid_set_size
    
        self.tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
        self.data_files = {"train": [], "valid": [], "test": []}
        self.dataset_filtered = None
        self.java_seq_generator = javaSequenceGenerator(cleanseq=cleanseq)

    def collect_files(self):
        for file in os.listdir(self.parsed_datapath):
            file_path = os.path.join(self.parsed_datapath, file)
            if "train" in file:
                self.data_files["train"].append(file_path)
            elif "valid" in file:
                self.data_files["valid"].append(file_path)
            elif "test" in file:
                self.data_files["test"].append(file_path)

    def filter_data(self, dataset):
        dataset_filtered = dataset.filter(
            lambda example: 
                            len(example['input_ids']) <= self.max_length
                            and len(example['labels']) <= self.max_length
                            and len(example["labels"]) > self.min_length # to be sure we have generated seq from our generator without any exception
                            , 
            num_proc=self.processor_number,
        )
        return dataset_filtered
    
    def preprocess_combined(self, examples):
        # Processing the 'code' field
        contents = examples["code"]
        model_inputs = self.tokenizer(
            contents,
        )
        # Processing the 'xml' field
        xmls = examples["xml"]
        seqs = [json.dumps(self.java_seq_generator.generate_sequence(xml)) for xml in xmls]
        labels = self.tokenizer(
            seqs,
        ).input_ids
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
            "seq": seqs,
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
            'repo', 'code_tokens', 'language', 'sha', 
            'url', 'path', 'original_string', 'partition', 'docstring', 
            'docstring_tokens', 'func_name'
        ]
        dataset_with_input = dataset_with_input.remove_columns(columns_to_remove)
        print(f"{dataset_with_input} processed.")
        self.dataset_filtered = self.filter_data(dataset_with_input)
        print(f"{self.dataset_filtered} processed.")

        self.dataset_filtered.save_to_disk(self.filtered_datapath)



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
    parser.add_argument("--python", action="store_true", help="prepare python dataset.")
    parser.add_argument("--java", action="store_true", help="prepare java dataset.")
    args = parser.parse_args()
    if args.java:
        print('staring java dataset preparation')
        processor = DatasetProcessorJava(parsedDatapath=os.path.join(parent_dir, config['parsedDataJava'].lstrip(os.sep)), 
                                        filteredDatapath=os.path.join(parent_dir, config['filteredDataJava'].lstrip(os.sep)),
                                        processorNumber=int(config['filterProcessNumber']),
                                        max_length=int(config['maxlength']),
                                        min_length=int(config['minlength']),
                                        cleanseq=args.cleanseq)
        processor.run()
    if args.python:
        print('starting python dataset preparation')
        processorN = int(config['filterProcessNumber'])
        processor = DatasetProcessorPython(decompressedPythonDatapath=os.path.join(parent_dir, config['decompressedPythonData'].lstrip(os.sep)), 
                                            filteredDatapath=os.path.join(parent_dir, config['filteredDataPython'].lstrip(os.sep)),
                                            processorNumber=processorN,
                                            max_length=int(config['maxlength']),
                                            min_length=int(config['minlength']),
                                            cleanseq=args.cleanseq)
        processor.run()

if __name__ == "__main__":
    main()
    





























# import json
# from datasets import load_from_disk, load_dataset
# from transformers import RobertaTokenizer
# from seq_generator_models.java_model.java_seq_generator import javaSequenceGenerator
# import os
# from dotenv import dotenv_values



# OUTPUT_PATH =  os.path.dirname(os.getcwd()) + dotenv_values("../.env")["OUTPUT_PATH"]
# DATASET_PATH = "../dataset/codesearchnet-java-discovered/"

# tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")

# data_files = {"train": [], "valid": [], "test": []}

# for file in os.listdir(DATASET_PATH):
#     file_path = os.path.join(DATASET_PATH, file)
#     if "train" in file:
#         data_files["train"].append(file_path)
#     elif "valid" in file:
#         data_files["valid"].append(file_path)
#     elif "test" in file:
#         data_files["test"].append(file_path)

# print(data_files)

# dataset = load_dataset("json", data_files=data_files)
# print(dataset)


# def preprocess_examples(examples):
#     contents = examples["code"]
#     model_inputs = tokenizer(contents)
#     return model_inputs


# dataset_with_input = dataset.map(
#     preprocess_examples,
#     batched=True,
#     batch_size=100,
#     num_proc=64,
# )


# def preprocess_examples(examples):
#     xmi = examples["xmi"]

#     seqs = [generate_sequence(xmi_string) for xmi_string in xmi]
#     seqs = [json.dumps(seq) for seq in seqs]

#     labels = tokenizer(seqs).input_ids

#     return {"seq": seqs, "labels": labels}


# dataset_with_output = dataset_with_input.map(
#     preprocess_examples,
#     batched=True,
#     batch_size=10,
#     num_proc=64,
# )

# print(dataset_with_output)
# dataset_with_output.save_to_disk(f"{OUTPUT_PATH}/dataset/seq_dataset")

# MAX_LENGTH = 505

# dataset_filtered = dataset_with_output.filter(
#     lambda example: len(example["input_ids"]) <= MAX_LENGTH, num_proc=64
# )
# print(dataset_filtered)
# dataset_filtered = dataset_filtered.filter(
#     lambda example: len(example["labels"]) <= MAX_LENGTH,
#     num_proc=64,
# )
# print(dataset_filtered)
# dataset_filtered = dataset_filtered.filter(
#     lambda example: len(example["seq"]) > 10,
#     num_proc=64,
# )
# print(dataset_filtered)

# print(dataset_filtered)
# dataset_filtered.save_to_disk(f"{OUTPUT_PATH}/dataset/seq_dataset_filtered")
