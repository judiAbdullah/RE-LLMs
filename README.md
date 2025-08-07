# Evaluation Artifact for LLM-Based Re-Engineering of Sequence Diagrams

## Overview

This repository contains the setup for our study _On the Generalization Capabilities of LLMs for Reverse Engineering Sequence Diagrams_ which is accepted at the 7th Workshop on Artificial Intelligence and Model-driven Engineering ([MDE Intelligence](https://mde-intelligence.github.io/)). 
In the paper, we finetune the LLM CodeT5 on the task of generating a sequence diagram representation for Java. 
We examine the transfer capabilities of the finetuned model by giving it the same task with Python methods as input.

This repository comprises all the source code used to  

- create the grund-truth dataset for Java and Python
- finetune codet5-small
- extract sequence diagrams for Python
- measure the similarity of a generated and a groundtruth diagram
- visualize sequence diagrams in dot-format and the evaluation results


__Authors__: Judi Abdullah, Sandra Greiner

## Project Contents

The project contains the following packages:

- [JavaParse](https://github.com/judiAbdullah/RE-LLMs/tree/main/JavaParse) is a Netbeans project that takes Java source code, parses it and transforms stores it in XML format. The XML files are transformed into JSON with the scripts in [seq_generator_models](https://github.com/judiAbdullah/RE-LLMs/tree/main/seq_generator_models)

- [dataset](https://github.com/judiAbdullah/RE-LLMs/tree/main/dataset) contains all data needed to start experiment and also the new generated data will be stored here
    - `java_dataset` in startion experiment should include `java/final/jsonl` which will have all data split in .json form
    - `python_dataset` in startion experiment should include `python/final/jsonl` which will have all data split in .json form

- [modelsData](https://github.com/judiAbdullah/RE-LLMs/tree/main/modelsData/seq_codet5_finetuned) contains the trained model checkpoints folder and also finetuned folder starting experiment this folder should be empty after traing we will have both folder created by training script

- [seq_generator_models](https://github.com/judiAbdullah/RE-LLMs/tree/main/seq_generator_models)
    - `python_model` folder where we have python sequence generator
    - `java_model` folder where we have java sequence generator

- [experiment_scripts](https://github.com/judiAbdullah/RE-LLMs/tree/main/experiment_scripts) contains all scripts that are needed to conduct our experiment

- [visualization](https://github.com/judiAbdullah/RE-LLMs/tree/main/visualization) contains scripts to plot sequence diagrams in the dot format


# Conduct the experiments

## Download Data

Download the [datasets and a finetuned version]((https://zenodo.org/records/16755656)) of codet5-small from Zenodo and place the data in the respective folders. 


## Requirements

This repository contains a Java project to transform Java methods into XML-files that hold their AST and Python scripts to perform the remaining experiment. 
Thus, the full experiment depends on Maven and Python libraries.

When using Conda you can find all necessary libraries in
- [environment.yml](https://github.com/judiAbdullah/RE-LLMs/blob/main/environment.yml) describes all necessary libs for running experiment. Add them to your environment in any way you prefere or use it to create a new environment (for conda environment)

### Python Dependencies

With classical Python install perform the following:

1. Create a Python Virtual Environment inside the package directory. This can be done as follows, but might be slightly different depending on your OS or Python distribution: 

    ```sh 
    python -m venv .venv
    ```

    Ensure that you use __Python 3.10__.

2. Activate your the created virtual environment:

    ```sh
        source .venv/bin/activate
    ```
3. Install the required Pyhton packages from the requirements.txt:

    ```sh
    pip install -r requirements.txt
    ```

__Please note__

The library tree-sittter needs to be integrated in version 0.21.3. This is done in the requirements file automatically.

```sh
pip install tree-sitter==0.21.3
```

### Java Dependencies

Ensure that `mvn` is installed on your system:

For Linux perform the following:

1) Download the sources for JDK 21
    
```sh
    wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.tar.gz
```
2) Integrate Java in your classpath
    - `mkdir -p ~/java`
    - `tar -xzf jdk-21_linux-x64_bin.tar.gz -C ~/java`
    - `export JAVA_HOME=~/java/jdk-21.0.5`
    - `export PATH=$JAVA_HOME/bin:$PATH`
    - `source ~/.bashrc`
    - `java -version`

3) Download the sources for Maven

    ```sh 
    wget https://dlcdn.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz
    ```
4) Integrate Maven in your classpath
    - `mkdir -p ~/maven`
    - `tar -xzf apache-maven-3.9.6-bin.tar.gz -C ~/maven`
    - `export MAVEN_HOME=~/maven/apache-maven-3.9.6`
    - `export PATH=$MAVEN_HOME/bin:$PATH`
    - `source ~/.bashrc`
    - `mvn -version`


## Run the experiment

1. Ensure that the following folders are not empty (and contain the necessary data for the experiment) 
    - [dataset/java_dataset](https://github.com/judiAbdullah/RE-LLMs/tree/main/dataset/java_dataset) needs to contain `(java/final/json/(test,train,valid))`
    - [dataset/python_dataset](https://github.com/judiAbdullah/RE-LLMs/tree/main/dataset/python_dataset) shall contain `(python/final/json/(test,train,valid))`

2. Run the script [decompress](https://github.com/judiAbdullah/RE-LLMs/blob/main/experiment_scripts/decompress.py) from the experiements_scripts directory

    `python decompress.py --java --python` 

    This extracts the data from the archived files and transforms them into json files. They will be stored in `dataset/[language]_dataset/decompressedData`

3. Run the Maven project from [JavaParse](https://github.com/judiAbdullah/RE-LLMs/tree/main/JavaParse). 

    It parses the entire Java dataset and stores the resulting AST in XML format 

    - be sure mvn is installed on your system
    - in the project folder `JavaParse` run 
        ```sh 
        mvn clean compile
        ```
    - run 
        ```sh 
        mvn exec:java
        ```
    
    The parsed data will be stored in `dataset/java_dataset/parsedData` as jsonl files. Please note, this step may take several minutes.

4. Conduct the training
    1. prepare the dataset by running the script [prepare_dataset.py](https://github.com/judiAbdullah/RE-LLMs/blob/main/experiment_scripts/prepare_dataset.py):

        `python prepare_dataset.py --java --python` 
        
        This creates the filtered set of sequence diagrams and prepares the data for training and testing the LLM.
        
        Please note: some samples are unparsable because of syntax error those will be excluded
    
    2. perform the fintuning with the script [finetune_codeT5.py](https://github.com/judiAbdullah/RE-LLMs/blob/main/experiment_scripts/finetune_codeT5.py):
         `python finetune_codeT5.py --train` 
    
5. compute the metrics _exact match_ and _CodeBLEU_:
        
    - evaluation (chose from below what you need) make sure the `CodeBLEU` folder exist in `experiment_scripts` folder
        
        - Java test dataset
            
            - run `python evaluate_script.py --generate --compute --javadata` to generate evaluation data and compute the evaluation score
            
            - run `python evaluate_script.py --compute --javadata` to compute the evaluation score if you only want score calculation if the data already generated

        - Python test dataset

            - run `python evaluate_script.py --generate --compute --pythondata` to generate evaluation data and compute the evaluation score
            
            - run `python evaluate_script.py --compute --pythondata` to compute the evaluation score if you only want score calculation if the data already generated
            
            - to remove (title element) from evaluation add `--titleremove` you can add it only with `--compute` if you have `--generate` done
        
    the generated data will be stored in `seq_dataset_filtered_eval`
    
6. compute the __structural similarity__ 
        
    - to generate the evaluation data and compute the evaluation score:
        
    ```sh 
    python specify_evaluation.py --generate --compute
    ``` 
    
    - to compute only the evaluation score if you have the data already generated, run

        ```sh 
        python specify_evaluation.py --compute
        ```

        - posible to specify which data to use `--pythondata` or `--javadata`
        
        - to add fixing json bracket if exist and solvable add `--fixjson`
        
        - generated data will be stored in `seq_dataset_filtered_evaluated`

7. optional:
        
    - run `python evaluate_store.py` to store evaluation as panda dataframe to visualize it easly
    
    - run show data to see samples of your data sets
        - `python showdata.py --dataset filteredDataJava --samples 5`
        - `python showdata.py --dataset filteredDataPython --samples 5`
        - `python showdata.py --dataset evalDatasetpathJava --samples 5`
        - `python showdata.py --dataset evalDatasetpathPython --samples 5`
    - run `python testcases.py` to evaluate the model using hand writen test cases
        - the result will be stored in `dataset/testcase.csv`
        - you can print the samples and show the evaluation in `evaluation.ipynb`

    - in `visualization` folder run `python generate_seq_diagram.py` or run `python testcase_draw.py` to plot testcases to show it visualy


## Some Issues and how they have been solved:

- everywhere, where you find (<b> seq </b>) this is a sequence diagram in JSON and (<b>seqs</b>) this is a list of sequence for list of functions

- in seq_drawer we added replacing ',' between function parameters with ' -' because the api cause error with having ',' in object name but we keep it sequence json

- python seq plot has problem in plot in case eg:{'a':"1", 'b': 2}.items() but seq in correct

- AST-unparsable samples seq_generation return special case of seq which is empty to be excluded in prapareData

- not possible to determine from ast which kw has default known kw args come after *args{
    ```
    def example_function(pos_only1, pos_only2, /, args1, args2, args3, *args, kw1, kw2=None, kw3, kw4, **kwargs):
    ```
    }

- special case: (args = defaultvalue => args=defaultvalue) fixed with regex

- exclude title element from seq from evaluation: (fixed for python by pass function code and extract the exact function def)

- exclude functions of subfunction call: excluded in prepare data using function defined in generator class

- special casse in python seq string element we have "str\"str" need to be changed to "str'str"

- newInstance in java is changed (from old version) to have the variable name included / we don't have this case in python since we can not distinguish between function call and object creation
    ```
    in java we create variable by type var; or type var = new tyep(); or type var = fun();
    in python we don't add type before the variable so the model doesn't detect newInstance for example var = fun()
    ```

- cleanseq : is a case we train and test the model in a certain situation which is: we remove all unused scopedVariable elements from sequence <b> this one has been entairly excluded from seq, so no need to use this args any more </b>
