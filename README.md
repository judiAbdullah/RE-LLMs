# LLM-Based Re-Engineering of Sequence Diagrams

## Auther: Judi Abdullah

## project structure

- `dataset` here we have our data folder all data needed to start experiment should be here and new generated data also stored here
    - `java_dataset` in startion experiment should include `java/final/jsonl` which will have all data split in .json form
    - `python_dataset` in startion experiment should include `python/final/jsonl` which will have all data split in .json form

- `modelsData` here we have the trained model checkpoints folder and also finetuned folder starting experiment this folder should be empty after traing we will have both folder created by training script

- `jupyter` folder where we have some jupyter files to test out code and data
    <br>those files are only to test the model over my test cases code
    - `model_generation_both_deterministic_creative.ipynb` contain full experiment for my test cases over deterministic and creative model generation parameters
    - `model_generation_creative.ipynb` contain full experiment for my test cases over creative model generation parameters
    - `model_generation_deterministic.ipynb` contain full experiment for my test cases over deterministic model generation parameters
    <br> here are other important file to take a look
    - `plots.ipynb` where is the plot of training progress loss and eval loss
    - `test_seq_generator_and_drawer.ipynb` where i do test for my seq generator models and also plot the seq as images
    - other file are not realy important

- `JavaParser` is netbeans project that will take the java data and parse it then add it to the dataset

- `seq_generator_models`
    - `python_model` folder where we have python sequence generator
    - `java_model` folder where we have java sequence generator

- `experiment_scripts` folder where we have all our experiment running scripts

- `visualization` contain some script to plot seq


## Requirements
 - `environment.yml` file inlcude all necessary libs for running experiment add them to your env in any way you prefere or use it to create a new env <b>note: conda used</b>

## needed lib version special case
 - pip install tree-sitter==0.21.3

 - not necessary 
    - create folder called cache
        `export HF_DATASETS_CACHE=<cachefolderpath>` <br>
        `source ~/.bashrc`

## Conduct/Repeat the experiments

1. be sure about existence of your data 
    - `(java/final/json/(test,train,valid))` in `dataset/java_dataset  ` folder
    - `(python/final/json/(test,train,valid))` in `dataset/python_dataset  ` folder
2. from `experiment_script` folder run `python decompress.py --java --python` -> extract data from zip to json files stor in `dataset/decompressedData`
3. run maven project from `JavaParser` to parse all java dataset and stor the AST as xml in the dataset 
    - be sure mvn is installed on your system
        - linux:
            - `wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.tar.gz`
            - `mkdir -p ~/java`
            - `tar -xzf jdk-21_linux-x64_bin.tar.gz -C ~/java`
            - `export JAVA_HOME=~/java/jdk-21.0.5`
            - `export PATH=$JAVA_HOME/bin:$PATH`
            - `source ~/.bashrc`
            - `java -version`
            - `wget https://dlcdn.apache.org/maven/maven-3/3.9.6/binaries/apache-maven-3.9.6-bin.tar.gz`
            - `mkdir -p ~/maven`
            - `tar -xzf apache-maven-3.9.6-bin.tar.gz -C ~/maven`
            - `export MAVEN_HOME=~/maven/apache-maven-3.9.6`
            - `export PATH=$MAVEN_HOME/bin:$PATH`
            - `source ~/.bashrc`
            - `mvn -version`
    - in project folder `JavaParser` run `mvn clean compile`
    - run `mvn exec:java`
    - parsed data will be stored in `dataset/java_dataset/parsedData` as json files
4. experiment scripts
    1. from `experiment_script` folder 
        - run `python prepare_dataset.py --java --python` it will generate sequence and prepare data for model training and testing
        <!-- not needed any more - add `--cleanseq` to clean you generated sequence as descriped previously -->
    2. note: some samples are unparsable because of syntax error those will be excluded
    3. run `python finetune_codeT5.py --train` to run training
    4. evaluation with CodeBLEU and exact match:
        - evaluation (chose from bellow what you need) make sure the `CodeBLEU` folder exist in `experiment_scripts` folder
            - Java test dataset
                - run `python evaluate_script.py --generate --compute --javadata` to generate evaluation data and compute the evaluation socre
                - run `python evaluate_script.py --compute --javadata` to compute the evaluation socre if you only want score calculation if the data already generated
            - Python test dataset
                - run `python evaluate_script.py --generate --compute --pythondata` to generate evaluation data and compute the evaluation socre
                - run `python evaluate_script.py --compute --pythondata` to compute the evaluation socre if you only want score calculation if the data already generated
                - to remove (title element) from evaluation add `--titleremove` you can add it only with --compute if you have --generate done
            - generated data will be stored in `seq_dataset_filtered_eval`
    
    5. evaluation with my specification if you generate in prestep then just compute here
        - run  `python specify_evaluation.py --generate --compute` to generate evaluation data and compute the evaluation socre
        - run  `python specify_evaluation.py --compute` to compute the evaluation socre if you only want score calculation if the data already generated
        - posible to specify which data to use `--pythondata` or `--javadata`
        generated
        - to add fixing json bracket if exist and solvable add `--fixjson`
        - generated data will be stored in `seq_dataset_filtered_evaluated`

    6. not necessary:
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

## notes for special cases spoted and how has been solved:

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
