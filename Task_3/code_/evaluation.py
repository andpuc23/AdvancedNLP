from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd
labels_list = ["'V'", "'_'", "'C-V'", "'ARGA'", "'ARG3'", "'ARG2'", "'ARG5'", "'ARG0'", "'ARG4'", "'ARG1'", "'C-ARG4'", "'C-ARG2'", "'R-ARG2'", "'C-ARG0'", "'R-ARG0'", "'C-ARG1'", "'C-ARG3'", "'R-ARG3'", "'R-ARG1'", "'ARGM-EXT'", "'ARGM-NEG'", "'ARGM-DIS'", "'ARGM-PRD'", "'ARGM-PRP'", "'ARGM-ADV'", "'ARGM-GOL'", "'ARGM-REC'", "'ARG1-DSP'", "'ARGM-MNR'", "'ARGM-PRR'", "'ARGM-DIR'", "'ARGM-LVB'", "'ARGM-TMP'", "'ARGM-ADJ'", "'ARGM-MOD'", "'ARGM-COM'", "'ARGM-LOC'", "'ARGM-CAU'", "'ARGM-CXN'", "'C-ARG1-DSP'", "'R-ARGM-DIR'", "'R-ARGM-LOC'", "'R-ARGM-ADV'", "'R-ARGM-MNR'", "'C-ARGM-EXT'", "'C-ARGM-LOC'", "'C-ARGM-CXN'", "'R-ARGM-ADJ'", "'R-ARGM-TMP'", "'R-ARGM-CAU'", "'C-ARGM-MNR'", "'R-ARGM-COM'"]

def class_report(file):
    labels_list = ["'V'", "'_'", "'C-V'", "'ARGA'", "'ARG3'", "'ARG2'", "'ARG5'", "'ARG0'", "'ARG4'", "'ARG1'", "'C-ARG4'", "'C-ARG2'", "'R-ARG2'", "'C-ARG0'", "'R-ARG0'", "'C-ARG1'", "'C-ARG3'", "'R-ARG3'", "'R-ARG1'", "'ARGM-EXT'", "'ARGM-NEG'", "'ARGM-DIS'", "'ARGM-PRD'", "'ARGM-PRP'", "'ARGM-ADV'", "'ARGM-GOL'", "'ARGM-REC'", "'ARG1-DSP'", "'ARGM-MNR'", "'ARGM-PRR'", "'ARGM-DIR'", "'ARGM-LVB'", "'ARGM-TMP'", "'ARGM-ADJ'", "'ARGM-MOD'", "'ARGM-COM'", "'ARGM-LOC'", "'ARGM-CAU'", "'ARGM-CXN'", "'C-ARG1-DSP'", "'R-ARGM-DIR'", "'R-ARGM-LOC'", "'R-ARGM-ADV'", "'R-ARGM-MNR'", "'C-ARGM-EXT'", "'C-ARGM-LOC'", "'C-ARGM-CXN'", "'R-ARGM-ADJ'", "'R-ARGM-TMP'", "'R-ARGM-CAU'", "'C-ARGM-MNR'", "'R-ARGM-COM'"]
    df = pd.read_csv(file)
    all_predictions = []
    all_gold = []
    predictions_list = df['prediction']
    gold_list = df['gold']
    for item in predictions_list:
        new_item = item.strip("[]").split(",")
        all_predictions.extend(new_item)

    for item in gold_list:
        new_item = item.strip("[]").split(",")
        all_gold.extend(new_item)
    unique_labels = sorted(set(all_gold + all_predictions))
    report = classification_report(all_gold, all_predictions, labels=labels_list)
    print(report)

def find_mistakes(file):
    df = pd.read_csv(file)
    wrong_predictions = df[df.iloc[:, -2] != df.iloc[:, -1]]
    wrong_predictions.to_csv('wrong_predictions_advanced.csv')
