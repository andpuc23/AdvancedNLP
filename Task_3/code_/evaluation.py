from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
import pandas as pd

baseline_file = '../data/output/base.csv'
advanced_file = '../data/output/advanced.csv'

def class_report_base(file):
    '''
    Since the file does not contain sentences without predicates, the last item of each list in the gold and 
    predicates file is removed. This item represented the predicate itself
    '''
    eval_labels_list = ["'V'", "'_'", "'C-V'", "'ARGA'", "'ARG3'", "'ARG2'", "'ARG5'", "'ARG0'", "'ARG4'", "'ARG1'", "'C-ARG4'", "'C-ARG2'", "'R-ARG2'", "'C-ARG0'", "'R-ARG0'", "'C-ARG1'", "'C-ARG3'", "'R-ARG3'", "'R-ARG1'", "'ARGM-EXT'", "'ARGM-NEG'", "'ARGM-DIS'", "'ARGM-PRD'", "'ARGM-PRP'", "'ARGM-ADV'", "'ARGM-GOL'", "'ARGM-REC'", "'ARG1-DSP'", "'ARGM-MNR'", "'ARGM-PRR'", "'ARGM-DIR'", "'ARGM-LVB'", "'ARGM-TMP'", "'ARGM-ADJ'", "'ARGM-MOD'", "'ARGM-COM'", "'ARGM-LOC'", "'ARGM-CAU'", "'ARGM-CXN'", "'C-ARG1-DSP'", "'R-ARGM-DIR'", "'R-ARGM-LOC'", "'R-ARGM-ADV'", "'R-ARGM-MNR'", "'C-ARGM-EXT'", "'C-ARGM-LOC'", "'C-ARGM-CXN'", "'R-ARGM-ADJ'", "'R-ARGM-TMP'", "'R-ARGM-CAU'", "'C-ARGM-MNR'", "'R-ARGM-COM'"]
    df = pd.read_csv(file)
    # all_predictions = [item.strip("[]").split(",")[:-1] for item in df['prediction']]
    # all_gold = [item.strip("[]").split(",")[:-1] for item in df['gold']]
    all_predictions = [item.strip("[]").split(",") for item in df['pred_restored']]
    all_gold = [item.strip("[]").split(",") for item in df['gold_restored']]


    all_predictions = [label.strip() for a_list in all_predictions for label in a_list]
    all_gold = [label.strip() for a_list in all_gold for label in a_list]
    report = classification_report(all_gold, all_predictions, labels=eval_labels_list)
    print(report)

def class_report_advanced(file):
    '''
    Since the file does not contain sentences without predicates, the last 3 item of each list in the gold and 
    predicates file is removed. This item represented the context, which always consisted of 3 elements.
    '''
    eval_labels_list = ["'V'", "'_'", "'C-V'", "'ARGA'", "'ARG3'", "'ARG2'", "'ARG5'", "'ARG0'", "'ARG4'", "'ARG1'", "'C-ARG4'", "'C-ARG2'", "'R-ARG2'", "'C-ARG0'", "'R-ARG0'", "'C-ARG1'", "'C-ARG3'", "'R-ARG3'", "'R-ARG1'", "'ARGM-EXT'", "'ARGM-NEG'", "'ARGM-DIS'", "'ARGM-PRD'", "'ARGM-PRP'", "'ARGM-ADV'", "'ARGM-GOL'", "'ARGM-REC'", "'ARG1-DSP'", "'ARGM-MNR'", "'ARGM-PRR'", "'ARGM-DIR'", "'ARGM-LVB'", "'ARGM-TMP'", "'ARGM-ADJ'", "'ARGM-MOD'", "'ARGM-COM'", "'ARGM-LOC'", "'ARGM-CAU'", "'ARGM-CXN'", "'C-ARG1-DSP'", "'R-ARGM-DIR'", "'R-ARGM-LOC'", "'R-ARGM-ADV'", "'R-ARGM-MNR'", "'C-ARGM-EXT'", "'C-ARGM-LOC'", "'C-ARGM-CXN'", "'R-ARGM-ADJ'", "'R-ARGM-TMP'", "'R-ARGM-CAU'", "'C-ARGM-MNR'", "'R-ARGM-COM'"]
    df = pd.read_csv(file)
    all_predictions = [item.strip("[]").split(",")[:-3] for item in df['prediction']]
    all_gold = [item.strip("[]").split(",")[:-3] for item in df['gold']]

    # all_predictions = [item.strip("[]").split(",") for item in df['pred_restored']]
    # all_gold = [item.strip("[]").split(",") for item in df['gold_restored']]

    all_predictions = [label.strip() for a_list in all_predictions for label in a_list]
    all_gold = [label.strip() for a_list in all_gold for label in a_list]
    report = classification_report(all_gold, all_predictions, labels=eval_labels_list)
    print(report)


# word_ids = [1,2,2,2,2,3,3,4]
# predictions = ['_', '_', 'ARG1', 'ARG2', 'ARG2', 'ARG1', 'ARG3', '_']

def shrink_predictions(word_ids, predictions):
    ids_and_preds = list(zip(word_ids, predictions))
    dict_ids_and_preds = {}
    final_dicts_ids_and_preds = {}
    for word_id, pred in ids_and_preds:
        if word_id in dict_ids_and_preds:
            dict_ids_and_preds[word_id].append(pred)
        else:
            dict_ids_and_preds[word_id] = [pred]
    for word_id, pred_list in dict_ids_and_preds.items():
        arguments_only_preds = []
        for element in pred_list:
            if element != '_':
                arguments_only_preds.append(element)
            if arguments_only_preds:
                most_common = max(arguments_only_preds, key=arguments_only_preds.count)
                final_dicts_ids_and_preds[word_id] = most_common
            else:
                final_dicts_ids_and_preds[word_id] = '_'
    return [dict_ids_and_preds[key][0] for key in sorted(dict_ids_and_preds.keys(), reverse=True)]
    # print(final_dicts_ids_and_preds)

# shrink_predictions(word_ids, predictions)


