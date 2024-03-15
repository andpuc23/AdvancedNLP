# AdvancedNLP
Task_2 consists of the code/data for the second assignment. 
Github link: https://github.com/andpuc23/AdvancedNLP/tree/main 

In main.ipynb, the data is imported, preprocessed, trained using Logistic Regression, and evaluated. There are 2 models that we have used in this project. The first being the identification model, which is used to identify whether or not tokens in a sentence are arguments. The other model is a classification model, which is used to label these arguments. 

You can run the notebook in its sequential order

We have chosen to train with a dataset of a slightly bigger size as the test set to reduce computation time. This consists of 30582 lines in the original dataset (en_ewt-up-train-small.conllu) and 143893 lines in read_conllu_train-small.csv. The smaller dataset can be found on Github!!!!

The original data (en_ewt-up-dev/train/test.conllu) is stored in UP-1.0 -> input. These files are the input of read_conllu_to_csv.py. The function in this file, read_conllu_write_csv, reads in the CONLL-U format file, extracts specific columns, and writes them into a CSV file. Each sentence in the original file is duplicated, based on how many predicates it has. 

The outputfile of read_conllu_write_csv is the inputfile of the function extract_features in extract_features.py.

The vectorizer for identification is a DictVectorizer, which is suitable for logistic regression. All features contain labels as values. The DictVectorizer takes a dictionary of feature mappings and transforms this into a matrix consisting of numerical values. This matrix can then be used as input for the Logistic Regression model.   

The vectorizor for classification is also a DictVectorizer, used for the same reasons as the identification model. Since the datasets for identification and classification slightly differ, there might be some differences in data vectorization. Therefore the vectorizors are different. 

The notebook itself does contain some comments to explasin lines that might not be clear. 


