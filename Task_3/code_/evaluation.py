from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd
eval_labels_list = ["'V'", "'_'", "'C-V'", "'ARGA'", "'ARG3'", "'ARG2'", "'ARG5'", "'ARG0'", "'ARG4'", "'ARG1'", "'C-ARG4'", "'C-ARG2'", "'R-ARG2'", "'C-ARG0'", "'R-ARG0'", "'C-ARG1'", "'C-ARG3'", "'R-ARG3'", "'R-ARG1'", "'ARGM-EXT'", "'ARGM-NEG'", "'ARGM-DIS'", "'ARGM-PRD'", "'ARGM-PRP'", "'ARGM-ADV'", "'ARGM-GOL'", "'ARGM-REC'", "'ARG1-DSP'", "'ARGM-MNR'", "'ARGM-PRR'", "'ARGM-DIR'", "'ARGM-LVB'", "'ARGM-TMP'", "'ARGM-ADJ'", "'ARGM-MOD'", "'ARGM-COM'", "'ARGM-LOC'", "'ARGM-CAU'", "'ARGM-CXN'", "'C-ARG1-DSP'", "'R-ARGM-DIR'", "'R-ARGM-LOC'", "'R-ARGM-ADV'", "'R-ARGM-MNR'", "'C-ARGM-EXT'", "'C-ARGM-LOC'", "'C-ARGM-CXN'", "'R-ARGM-ADJ'", "'R-ARGM-TMP'", "'R-ARGM-CAU'", "'C-ARGM-MNR'", "'R-ARGM-COM'"]

baseline_file = '../data/output/base.csv'
advanced_file = '../data/output/advanced.csv'

def class_report_base(file, labels_list):
    '''
    Since the file does not contain sentences without predicates, the last item of each list in the gold and 
    predicates file is removed. This item represented the predicate itself
    '''
    df = pd.read_csv(file)
    all_predictions = [item.strip("[]").split(",")[:-1] for item in df['prediction']]
    all_gold = [item.strip("[]").split(",")[:-1] for item in df['gold']]
    all_predictions = [label.strip() for a_list in all_predictions for label in a_list]
    all_gold = [label.strip() for a_list in all_gold for label in a_list]
    report = classification_report(all_gold, all_predictions, labels=labels_list)
    print(report)

def class_report_advanced(file, labels_list):
    '''
    Since the file does not contain sentences without predicates, the last 3 item of each list in the gold and 
    predicates file is removed. This item represented the context, which always consisted of 3 elements.
    '''
    df = pd.read_csv(file)
    all_predictions = [item.strip("[]").split(",")[:-3] for item in df['prediction']]
    all_gold = [item.strip("[]").split(",")[:-3] for item in df['gold']]
    all_predictions = [label.strip() for a_list in all_predictions for label in a_list]
    all_gold = [label.strip() for a_list in all_gold for label in a_list]
    report = classification_report(all_gold, all_predictions, labels=labels_list)
    print(report)


