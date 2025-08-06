import gzip
import shutil
import os
from pathlib import Path
from dotenv import dotenv_values
import argparse


def decompress(java=False, python=False):
    config = dotenv_values("../.env")
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    if java:
        source_java_folder = os.path.join(parent_dir, config['purJavaData'].lstrip(os.sep))
        source_java_output_folder = os.path.join(parent_dir, config['decompressedJavaData'].lstrip(os.sep))
        print("java sorce folder:",source_java_folder)
        print("java out folder:", source_java_output_folder)
        # Ensure the output directory exists
        os.makedirs(source_java_output_folder, exist_ok=True)

        # Loop through all files in the source folder
        for root, dirs, files in os.walk(source_java_folder):
            for filename in files:
                if filename.endswith('.jsonl.gz'):
                    input_file_path = os.path.join(root, filename)
                    output_file_path = os.path.join(source_java_output_folder, filename[:-3])  # Remove .gz extension

                    # Decompress the file
                    with gzip.open(input_file_path, 'rb') as f_in:
                        with open(output_file_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    print(f'Decompressed file saved to: {output_file_path}')
        print("All Java data files has been extracted")
    if python:
        source_python_folder = os.path.join(parent_dir, config['purPythonData'].lstrip(os.sep))
        source_python_output_folder = os.path.join(parent_dir, config['decompressedPythonData'].lstrip(os.sep))
        print("python sorce folder:",source_python_folder)
        print("python out folder:", source_python_output_folder)
        # Ensure the output directory exists
        os.makedirs(source_python_output_folder, exist_ok=True)

        # Loop through all files in the source folder
        for root, dirs, files in os.walk(source_python_folder):
            for filename in files:
                if filename.endswith('.jsonl.gz'):
                    input_file_path = os.path.join(root, filename)
                    output_file_path = os.path.join(source_python_output_folder, filename[:-3])  # Remove .gz extension

                    # Decompress the file
                    with gzip.open(input_file_path, 'rb') as f_in:
                        with open(output_file_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    print(f'Decompressed file saved to: {output_file_path}')
        print("All Python data files has been extracted")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model generation and compute metrics.")
    parser.add_argument("--java", action="store_true", help="Run java file decompress")
    parser.add_argument("--python", action="store_true", help="Run python files decompress")
    args = parser.parse_args()
    decompress(args.java, args.python)

if __name__ == '__main__':
    main()