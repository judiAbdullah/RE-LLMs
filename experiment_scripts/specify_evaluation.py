import json
from pprint import pprint
from datasets import load_from_disk, load_metric, DatasetDict
import pandas as pd
import evaluate
import numpy as np
import scipy.stats as stats
from transformers import (
    DataCollatorForSeq2Seq,
    RobertaTokenizer,
    T5ForConditionalGeneration,
)
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from dotenv import dotenv_values
import os
import sys
import argparse
from collections import Counter

import jsondiff
from jsondiff.symbols import Symbol

from collections import defaultdict
from CodeBLEU.calc_code_bleu import compute_code_bleu



parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)

config = dotenv_values("../.env")
finetunedmodelpath = os.path.join(parent_dir, config['modelDataFinetuned'].lstrip(os.sep))
num_workers = int(config['num_workers'])
BATCH_SIZE = int(config['BATCH_SIZE'])



def evalGenerate(filteredDatapath, evaldatapath, data='java'):
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    model = T5ForConditionalGeneration.from_pretrained(finetunedmodelpath)
    device = 'cuda:0'
    if data == 'java':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif data == 'python':
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    model_gpu = model.to(device)

    datasetorigin = load_from_disk(filteredDatapath)
    # print(datasetorigin)
    columns_to_remove = ["code", "xml", "seq", "origin_seq"]
    existing_columns = [col for col in columns_to_remove if col in datasetorigin.column_names['test']]
    # print("existing column", existing_columns)
    dataset = datasetorigin.remove_columns(existing_columns)
    # print(dataset)
    

    data_collator = DataCollatorForSeq2Seq(tokenizer)
    eval_sampler = SequentialSampler(dataset["test"])
    eval_dataloader = DataLoader(
        dataset["test"],
        sampler=eval_sampler,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,  # Ensure num_workers in .env file is appropriate for your machine
        pin_memory=True,
        collate_fn=data_collator,
        shuffle=False
    )

    gen_outputs  = []
    model_gpu.eval()
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        source_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            outputs = model_gpu.generate(
                source_ids,

                # combination of deterministic and creative generation parameters
                do_sample=True,
                max_length=512,
                num_beams=5,
                temperature=0.3,
                top_k=50,
                top_p=0.8,

                # deterministic generation parameters
                # do_sample=False,
                # max_length=512,
                # num_beams=5,
                # early_stopping=True

                # creative generation parameters
                # do_sample=True,
                # max_length=512
                # temperature=0.7,  # 0.7-1.0 often works better for creative output
                # top_k=50,
                # top_p=0.9,
                
            )
            gen_outputs.extend(outputs.cpu().tolist())
    
    print('start decoding generated sequences')
    gen_dec = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False,) for output in tqdm(gen_outputs,  total=len(gen_outputs))]

    print("done generating")

    dataset_eval = datasetorigin["test"].add_column("generated", gen_outputs)
    dataset_eval = dataset_eval.add_column("generated_decoded", gen_dec)
    print("saving")
    dataset_eval.save_to_disk(evaldatapath)
    return dataset_eval




class JsonBasedEvaluation:
    def __init__(self, fixjson=False):
        self.fixjson = fixjson

    def fix_json(seq):
        def matchBracket(current, stack):
            if ((current == ']' and stack == '[') or 
                (current == '}' and stack == '{')):
                return True
            return False
        def getClose(stack):
            if stack == '[':
                return ']'
            elif stack == '{':
                return '}'
        stack = []
        for i, c in enumerate(list(seq)):
            if c == "[":
                stack.append("[")
            elif c == "{":
                stack.append("{") 
            elif c == "]" or c == "}":
                if len(stack) <= 0 or not matchBracket(c, stack[-1]):
                    raise 'error in open bracket'
                if matchBracket(c, stack[-1]):
                    stack.pop()
        while len(stack) > 0:
            seq += getClose(stack[-1])
            stack.pop()                    
        return seq

    def count_nested_dicts(self, d):
        """Recursively count the number of dictionaries within a dictionary, including the main one."""
        count = 1  # Start with the main dictionary itself
        if isinstance(d, dict):
            # Iterate through all key-value pairs and count nested dictionaries
            for value in d.values():
                if isinstance(value, dict):
                    count += self.count_nested_dicts(value)  # Recursively count nested dictionaries
                elif isinstance(value, list):
                    # If the value is a list, check each item
                    for item in value:
                        if isinstance(item, dict):
                            count += self.count_nested_dicts(item)
        return count

    def convert_quotes(self, value):
        """Convert quotes such that the outer quotes are single quotes and inner quotes are double quotes."""
        if isinstance(value, str):
            # Replace any inner double quotes with single quotes and wrap the string in single quotes.
            value = value.replace('"', "'")  # Replace inner double quotes with single quotes
            # Wrap the value in single quotes (outer quotes)
            value = f"'{value}'"
            return value
        elif isinstance(value, list):
            return [self.convert_quotes(item) for item in value]
        elif isinstance(value, dict):
            return {key: self.convert_quotes(val) for key, val in value.items()}
        return value

    def make_hashable(self, obj):
        """Recursively convert lists to tuples so the object becomes hashable."""
        if isinstance(obj, list):
            return tuple(self.make_hashable(x) for x in obj)
        if isinstance(obj, dict):
            return frozenset((k, self.make_hashable(v)) for k, v in obj.items())
        return obj
    
    def extract_sub_dicts(self, di, path=None):
        if path is None:
            path = []
        dicts = []
        basedict = {}
        for key, value in di.items():
            if isinstance(value, list) and len(value) != 0 and isinstance(value[0], dict):
                basedict[key] = []
                for index, sub_dict in enumerate(value):
                    new_path = path.copy()
                    new_path.append(key)
                    new_path.append(index)
                    dicts.extend(self.extract_sub_dicts(sub_dict, new_path))
            else:
                basedict[key] = value
        dicts.append((path, basedict))
        return dicts

    def compare_2_paths(self, p1, p2):
        """
        return 
        1: if same
        0: if changes in string element
        2: if changes in integer element
        """
        isSame = 1
        if len(p1) != len(p2):
            return 0
        if p1 == p2:
            return 1
        for i in range(len(p1)):
            if isinstance(p1[i], str) and isinstance(p2[i], str):
                if p1[i] != p2[i]:
                    return 0
            elif isinstance(p1[i], int) and isinstance(p2[i], int):
                if p1[i] != p2[i]:
                    isSame = 2
            else:
                return 0
        return isSame
    
    def compare_2_dicts(self, d1, d2):
        """
        return 
        1: if same
        0: if change in keys
        2: if change in values
        """
        isSame = 1
        if d1 == d2:
            return 1
        d1keys = list(d1.keys())
        d2keys = list(d2.keys())
        if d1keys != d2keys:
            return 0
        return 2

    def compare_sub_dicts(self, ref, gen):
        exact_matches = 0
        deleted_els = 0
        inserted_els = 0
        position_m = 0
        content_m = 0
        seprated_dicts = {
            'exact_matches': [],
            'deleted_els': [],
            'inserted_els': [],
            'position_m': [],
            'content_m': []
        }
        # same element with same path
        indices_to_delete = []
        for index, (pr, dr) in enumerate(ref):
            temp_gen = []
            readchedIndex = -1
            for i, (pg, dg) in enumerate(gen):
                d_comp = self.compare_2_dicts(dr, dg)
                p_comp = self.compare_2_paths(pr, pg)
                if (p_comp == 1 or p_comp == 2) and d_comp == 1:
                    seprated_dicts['exact_matches'].append((pr.copy(), dr.copy()))
                    indices_to_delete.append(index)
                    readchedIndex = i
                    break
            if readchedIndex != -1:
                del gen[readchedIndex]
        indices_to_delete.sort(reverse=True)
        for index in indices_to_delete:
            del ref[index]

        # same content diff path
        indices_to_delete = []
        for index, (pr, dr) in enumerate(ref):
            temp_gen = []
            readchedIndex = -1
            for i, (pg, dg) in enumerate(gen):
                d_comp = self.compare_2_dicts(dr, dg)
                p_comp = self.compare_2_paths(pr, pg)
                if dr == dg:
                    seprated_dicts['position_m'].append((pr.copy(), pg.copy(), dr.copy()))
                    indices_to_delete.append(index)
                    readchedIndex = i
                    break
                else:
                    temp_gen.append((pg.copy(), dg.copy()))
            if readchedIndex != -1:
                del gen[readchedIndex]
        indices_to_delete.sort(reverse=True)
        for index in indices_to_delete:
            del ref[index]
        
        # same path diff content
        indices_to_delete = []
        for index, (pr, dr) in enumerate(ref):
            temp_gen = []
            readchedIndex = -1
            for i, (pg, dg) in enumerate(gen):
                d_comp = self.compare_2_dicts(dr, dg)
                p_comp = self.compare_2_paths(pr, pg)
                if p_comp == 1:
                    if d_comp == 2:
                        seprated_dicts['content_m'].append((pr.copy(), dr.copy(), dg.copy()))
                        indices_to_delete.append(index)
                        readchedIndex = i
                        break
                else:
                    temp_gen.append((pg.copy(), dg.copy()))
            if readchedIndex != -1:
                del gen[readchedIndex]
        indices_to_delete.sort(reverse=True)
        for index in indices_to_delete:
            del ref[index]
        
        if len(ref) != 0:
            for el in ref:
                seprated_dicts['deleted_els'].append(el)
        if len(gen) != 0:
            for el in gen:
                seprated_dicts['inserted_els'].append(el)

        exact_matches = len(seprated_dicts['exact_matches'])
        deleted_els = len(seprated_dicts['deleted_els'])
        inserted_els = len(seprated_dicts['inserted_els'])
        content_m = len(seprated_dicts['content_m'])
        position_m = len(seprated_dicts['position_m'])
        return exact_matches, deleted_els, inserted_els, content_m, position_m, seprated_dicts

    def calculate_generation_metrics_classification(self, ref_num, gen_num, exact_match, num_changes, num_deleted, num_inserted, change_position):
        """
        Calculates classification metrics for comparing two dictionaries (reference and generated) based on their differences.

        The metrics calculated include:
        - True Positives (TP): Number of elements correctly classified as unchanged.
        - True Negatives (TN): Number of elements correctly classified as inserted.
        - False Positives (FP): Number of elements incorrectly classified as changed.
        - False Negatives (FN): Number of elements incorrectly classified as deleted.
        - Accuracy: The proportion of correctly classified elements out of all elements.
        - Precision: The proportion of true positive changes out of all changes (both true positive and false positive).
        - Recall: The proportion of true positive changes out of all actual changes (true positive and false negative).
        - Error Rate: The proportion of incorrectly classified elements (false positives and false negatives).

        Args:
            ref_num (int): The total number of elements in the reference dictionary.
            gen_num (int): The total number of elements in the generated dictionary.
            num_changes (int): The number of elements that were changed between the reference and generated dictionaries.
            num_deleted (int): The number of elements that were deleted in the generated dictionary.
            num_inserted (int): The number of elements that were inserted into the generated dictionary.
            change_position (int): The number of changes that are position-matched between the reference and generated dictionaries.

        Returns:
            dict: A dictionary containing the following metrics:
                - "accuracy": The accuracy of the classification.
                - "precision": The precision of the classification.
                - "recall": The recall of the classification.
                - "error_rate": The error rate of the classification.
                - "TP": The number of true positive elements.
                - "TN": The number of true negative elements.
                - "FP": The number of false positive elements.
                - "FN": The number of false negative elements.
        """

        TP = exact_match
        TN = num_inserted
        FP = (num_changes + change_position)
        FN = num_deleted


        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        error_rate = (FP + FN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "error_rate": error_rate,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        }

    def calculate_generation_groundTruth_distance(self, ref_num, gen_num, exact_match, num_changes, num_deleted, num_inserted, change_position):
        """
        Calculates distance metrics for comparing two dictionaries (reference and generated) based on their differences.

        The metrics calculated:
            cost weight:
                - inserts and deletes have cost 1 (result from no match at all), 
                - move and modifications have cost 0.5 (they result from partially correct matches)
            - edit distance then is calculated as No- of inserts + No. of deletes + 0.5 No. of moves + 0.5 No. of modifications
            - relative edit distance (normalized to [0.0 .. 1.0] is calculated as: edit distance / No of. elements Gen + No of. elements GT

        Args:
            ref_num (int): The total number of elements in the reference dictionary.
            gen_num (int): The total number of elements in the generated dictionary.
            exact_match (int): The number of elements that exactly match between the reference and generated dictionaries.
            num_changes (int): The number of elements that were modified between the reference and generated dictionaries.
            num_deleted (int): The number of elements that were deleted in the generated dictionary.
            num_inserted (int): The number of elements that were inserted into the generated dictionary.
            change_position (int): The number of changes that are position-changed between the reference and generated dictionaries.

        Returns:
            distance (float): A number between [0, 1] that represent the distance between reference dict and generated dict
        """

        return (0.5 * num_changes + 0.5 * change_position + num_inserted + num_deleted) / (gen_num + ref_num)

    def evaluate_sample(self, ref, gen):
        """
        Evaluates the difference between a reference and a generated object and calculates various metrics.

        Args:
            ref (dict): The reference object (could be a dictionary or nested structure).
            gen (dict): The generated object (could be a dictionary or nested structure).

        Returns:
            dict: A dictionary containing the following evaluation metrics:
                - 'num_of_ref_ele': The number of elements in the reference object.
                - 'num_of_gen_ele': The number of elements in the generated object.
                - 'num_of_changed': The number of changed elements.
                - 'num_of_deleted': The number of deleted elements.
                - 'num_of_inserted': The number of inserted elements.
                - 'num_of_replaced': The number of replaced (position-matched) elements.
        """
        
        ref = self.convert_quotes(ref)
        gen = self.convert_quotes(gen)
        ref = self.extract_sub_dicts(ref)
        gen = self.extract_sub_dicts(gen)
        ref_num = len(ref)
        gen_num = len(gen)


        exact_matches, num_deleted, num_inserted, num_changes, change_position, separated_elements = self.compare_sub_dicts(ref=ref, gen=gen)

        distance_ref_2_gen = self.calculate_generation_groundTruth_distance(ref_num, gen_num, exact_matches, num_changes, num_deleted, num_inserted, change_position)
        eval = {
            'num_of_ref_ele': ref_num,
            'num_of_gen_ele': gen_num,
            'num_of_changed': num_changes,
            'num_of_deleted': num_deleted,
            'num_of_inserted': num_inserted,
            'num_of_replaced': change_position,
            'distance_metricies': distance_ref_2_gen
            # 'separated':separated_elements
        }
        return eval
    
    def compare_dicts(self, ref, gen):
        """
        Compare two lists of tuples containing content and position.

        Parameters:
            ref (list): The first list of tuples (content, position). groundtruth
            gen (list): The second list of tuples (content, position). generated

        Returns:
            dict: A dictionary containing accuracy, precision, and error percentages.
        """
        try:
            if type(ref) is not dict:
                parsedref = json.loads(ref)
            else:
                parsedref = ref
            if type(gen) is not dict:
                parsedgen = json.loads(gen)
            else:
                parsedgen = gen
        except Exception as e:
            if self.fix_json:
                try:
                    # print("fixjson is true")
                    fixedgen = self.fix_json(gen)
                    parsedgen = json.loads(fixedgen)
                except Exception as e:
                    # import traceback
                    # traceback.print_exc()
                    return {
                        "state":0, # means the sequence is not json form parsable even after solving bracket mistak
                        "eval": None
                    }
            else:
                return {
                    "state":0, # means the sequence is not json form parsable even after solving bracket mistak
                    "eval":None
                }
            
        parsedref
        parsedgen
        eval = self.evaluate_sample(parsedref, parsedgen)
        # add you code here
        return {
            "state":1,
            "eval": eval
        }

def evalCompute(dataset_eval, evaldatedapath, fixjson=False):
    metric_exactmatch = evaluate.load("exact_match")
    res = metric_exactmatch.compute(predictions=dataset_eval["generated_decoded"], references=dataset_eval["seq"])
    print("Exact match score:", res["exact_match"])
    res = compute_code_bleu(
        ref=dataset_eval["seq"],
        hyp=dataset_eval["generated_decoded"],
        lang="json",
        params=[1 / 3, 1 / 3, 1 / 3, 0],  # no dataflow information
    )
    print("CodeBLEU score:", res["code_bleu_score"])

    # ---------------------------------------------------------------------------------
    evaluator = JsonBasedEvaluation(fixjson)
    def compute_eval(example):
        seqref = example["seq"]
        seqgen = example["generated_decoded"]
        return {"eval": evaluator.compare_dicts(seqref, seqgen)}

    # Use the map method to apply the function
    updated_dataset = dataset_eval.map(
        compute_eval, 
        batched=False,
        num_proc=60,)
    
    # Save the updated dataset if needed
    updated_dataset.save_to_disk(evaldatedapath)


    eval_data = updated_dataset['eval']
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(eval_data)
    # Count the number of rows with state 0 and state 1
    state_counts = df['state'].value_counts()
    # Filter rows with state 1
    state_1_rows = df[df['state'] == 1].copy()
    # Extract distance_metricies as a Series
    distances = state_1_rows["eval"].apply(lambda x: x["distance_metricies"])

    # Compute stats
    mean_distance = distances.mean()
    median_distance = distances.median()
    std_distance = distances.std(ddof=1)

    # 95% confidence interval for the mean
    n = len(distances)
    if n > 1:
        # Standard error
        sem = std_distance / np.sqrt(n)
        # t-distribution critical value
        ci_range = stats.t.ppf(0.975, df=n-1) * sem
        ci_lower = mean_distance - ci_range
        ci_upper = mean_distance + ci_range
    else:
        ci_lower, ci_upper = (np.nan, np.nan)

    # Print results
    print(f"No- JSON-unParsable samples: {state_counts.get(0, 0)}")
    print(f"No- JSON-Parsable samples: {state_counts.get(1, 0)}")
    print(f"Average Distance: {mean_distance:.4f}")
    print(f"Median Distance: {median_distance:.4f}")
    print(f"Standard Deviation: {std_distance:.4f}")
    if n > 1:
        print(f"95% Confidence Interval for Mean: ({ci_lower:.4f}, {ci_upper:.4f})")
    else:
        print("95% Confidence Interval for Mean: not applicable (n<=1)")

    



def main():
    parser = argparse.ArgumentParser(description="Evaluate model generation and compute metrics.")
    parser.add_argument("--generate", action="store_true", help="Run evalGenerate to generate data.")
    parser.add_argument("--compute", action="store_true", help="Run evalCompute to compute metrics.")
    parser.add_argument("--fixjson", action="store_true", help="Fix JSON bracket before computing")
    parser.add_argument("--javadata", action="store_true", help="Run evaluation on java test data")
    parser.add_argument("--pythondata", action="store_true", help="Run evaluation on python test data")
    args = parser.parse_args()
    filteredDatapath = None
    evaldatapath = None
    evaldatedapath = None
    data = None
    if args.javadata:
        filteredDatapath = os.path.join(parent_dir, config['filteredDataJava'].lstrip(os.sep))
        evaldatapath=os.path.join(parent_dir, config['evalDatasetpathJava'].lstrip(os.sep))
        evaldatedapath=os.path.join(parent_dir, config['evaluatedDatasetpathJava'].lstrip(os.sep))
        data = 'java'
    else:
        filteredDatapath = os.path.join(parent_dir, config['filteredDataPython'].lstrip(os.sep))
        evaldatapath=os.path.join(parent_dir, config['evalDatasetpathPython'].lstrip(os.sep))
        evaldatedapath=os.path.join(parent_dir, config['evaluatedDatasetpathPython'].lstrip(os.sep))
        data = 'python'
    if args.generate:
        dataset_eval = evalGenerate(filteredDatapath, evaldatapath, data=data)
    if args.compute:
        if not os.path.exists(evaldatapath):
            print("Error: evaldatapath does not exist. Run with --generate first.")
            sys.exit(1)
        dataset_eval = load_from_disk(evaldatapath)
        evalCompute(dataset_eval, evaldatedapath, fixjson=args.fixjson)
    else:
        print("Error: Specify at least one of --generate or --compute or --dataset.")
        parser.print_help()

if __name__ == '__main__':
    main()

