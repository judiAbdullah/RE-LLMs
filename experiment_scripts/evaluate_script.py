from datasets import load_from_disk, load_metric, DatasetDict
from transformers import (
    DataCollatorForSeq2Seq,
    RobertaTokenizer,
    T5ForConditionalGeneration,
)
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import evaluate
from CodeBLEU.calc_code_bleu import compute_code_bleu
from dotenv import dotenv_values
import os
import numpy as np
import sys
import argparse
import re




parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)

config = dotenv_values("../.env")
finetunedmodelpath = os.path.join(parent_dir, config['modelDataFinetuned'].lstrip(os.sep))
num_workers = int(config['num_workers'])
BATCH_SIZE = int(config['BATCH_SIZE'])


def remove_title_from_json(json_str):
    try:
        # Regex to match `"title": "value"` where value can be any string
        updated_str = re.sub(r'"title"\s*:\s*".*?"\s*,?', '', json_str)
        
        # Clean up trailing commas after removing a field
        updated_str = re.sub(r',\s*}', '}', updated_str)
        updated_str = re.sub(r',\s*]', ']', updated_str)
        
        return updated_str
    except Exception as e:
        print(f"Error processing JSON: {e}")
        return json_str  # Return the original string in case of an error
    
def transform_title_remove(sample):
    # Replace this with your transformation logic
    sample["generated_decoded"] = remove_title_from_json(sample["generated_decoded"])
    sample["seq"] = remove_title_from_json(sample["seq"])
    return sample


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


def evalCompute(dataset_eval, titleremove=False):
    if titleremove:
        dataset_eval = dataset_eval.map(transform_title_remove)
    print('computing started')
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate model generation and compute metrics.")
    parser.add_argument("--generate", action="store_true", help="Run evalGenerate to generate data.")
    parser.add_argument("--compute", action="store_true", help="Run evalCompute to compute metrics.")
    parser.add_argument("--javadata", action="store_true", help="Run evaluation on java test data")
    parser.add_argument("--pythondata", action="store_true", help="Run evaluation on python test data")
    parser.add_argument("--titleremove", action="store_true", help="Run evaluation without 'title':'functiontitle(args)'")
    args = parser.parse_args()
    if args.generate and args.javadata:
        filteredDatapath = os.path.join(parent_dir, config['filteredDataJava'].lstrip(os.sep))
        evaldatapath=os.path.join(parent_dir, config['evalDatasetpathJava'].lstrip(os.sep))
        dataset_eval = evalGenerate(filteredDatapath, evaldatapath, data='java')
        if args.compute:
            evalCompute(dataset_eval, titleremove=args.titleremove)
    elif args.compute and args.javadata:
        evaldatapath=os.path.join(parent_dir, config['evalDatasetpathJava'].lstrip(os.sep))
        if not os.path.exists(evaldatapath):
            print("Error: evaldatapath does not exist. Run with --generate first.")
            sys.exit(1)
        dataset_eval = load_from_disk(evaldatapath)
        evalCompute(dataset_eval, titleremove=args.titleremove)
    elif args.generate and args.pythondata:
        filteredDatapath = os.path.join(parent_dir, config['filteredDataPython'].lstrip(os.sep))
        evaldatapath=os.path.join(parent_dir, config['evalDatasetpathPython'].lstrip(os.sep))
        dataset_eval = evalGenerate(filteredDatapath, evaldatapath, data='python')
        if args.compute:
            evalCompute(dataset_eval, titleremove=args.titleremove)
    elif args.compute and args.pythondata:
        evaldatapath=os.path.join(parent_dir, config['evalDatasetpathPython'].lstrip(os.sep))
        if not os.path.exists(evaldatapath):
            print("Error: evaldatapath does not exist. Run with --generate first.")
            sys.exit(1)
        dataset_eval = load_from_disk(evaldatapath)
        evalCompute(dataset_eval, titleremove=args.titleremove)
    else:
        print("Error: Specify at least one of --generate or --compute.")
        parser.print_help()

if __name__ == '__main__':
    main()

