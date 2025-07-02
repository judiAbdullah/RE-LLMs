import json
import pandas as pd
from datasets import load_from_disk
from CodeBLEU.calc_code_bleu import compute_code_bleu
import os
import sys
from dotenv import dotenv_values

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)
config = dotenv_values("../.env")



def store_evaluation_score_samples(dataset_dict, subset='test', n=10, start=0):
    """
    Visualize n samples from a specified subset of the dataset.

    Args:
        dataset_dict (DatasetDict): The dataset containing train, valid, and test subsets.
        subset (str): The subset to visualize ('train', 'valid', or 'test').
        n (int): Number of samples to display.
    """
    # if subset not in dataset_dict:
    #     raise ValueError(f"Subset '{subset}' not found in the dataset. Choose from: {list(dataset_dict.keys())}")
    # Select the subset
    # subset_data = dataset_dict[subset]
    subset_data = dataset_dict
    # Ensure n does not exceed the number of rows
    n = min(n, len(subset_data))
    # Sample the data
    sampled_data = subset_data.select(range(start, start + n))
    seqs = [json.dumps(x, indent=4) for x in sampled_data['seq']]
    codes = [x for x in sampled_data['code']]
    generated_decoded = [json.dumps(x, indent=4) for x in sampled_data['generated_decoded']]
    evalcmpute = []
    for i, x in enumerate(sampled_data):
        try:
            evalcmpute.append(compute_code_bleu(ref=[x["seq"]],
                                                hyp=[x["generated_decoded"]],
                                                lang="json",
                                                params=[1 / 3, 1 / 3, 1 / 3, 0]
                                            ))
        except AssertionError as e:
            evalcmpute.append("error in cmpute_code_bleu")
            print("error in:",i)
            
    # Prepare a DataFrame for textual columns
    df = pd.DataFrame({
        'Code': codes,
        'Seq': seqs,
        'Generated Seq': generated_decoded,
        'Evaluation': evalcmpute
    })
    return df


if __name__ == "__main__":
    evalDatasetpathJava = os.path.join(parent_dir, config['evalDatasetpathJava'].lstrip(os.sep))
    dataset = load_from_disk(evalDatasetpathJava)
    # Visualize samples
    df = store_evaluation_score_samples(dataset, n=10000, start=0)
    # Save the DataFrame to a CSV file
    output_file = os.path.join(parent_dir, config['evalstorejava'].lstrip(os.sep))
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


    evalDatasetpathPython = os.path.join(parent_dir, config['evalDatasetpathPython'].lstrip(os.sep))
    dataset = load_from_disk(evalDatasetpathPython)
    # Visualize samples
    df = store_evaluation_score_samples(dataset, n=10000, start=0)
    # Save the DataFrame to a CSV file
    output_file = os.path.join(parent_dir, config['evalstorepython'].lstrip(os.sep))
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")