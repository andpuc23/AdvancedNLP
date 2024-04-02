from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns

def class_report(file):
    '''
    Since the file does not contain sentences without predicates, the last item of each list in the gold and 
    predicates file is removed. This item represented the predicate itself
    Prints the classification report; saves the confusion matrix figure in the same folder
    :param file:str path to file with predictions, should be found in data/output
    '''
    eval_labels_list = ["'ARG0'", "'ARG1'", "'ARG1-DSP'", "'ARG2'", "'ARG3'", "'ARG4'", "'ARG5'", "'ARGA'", "'ARGM-ADJ'", "'ARGM-ADV'", "'ARGM-CAU'", "'ARGM-COM'", "'ARGM-CXN'", "'ARGM-DIR'", "'ARGM-DIS'", "'ARGM-EXT'", "'ARGM-GOL'", "'ARGM-LOC'", "'ARGM-LVB'", "'ARGM-MNR'", "'ARGM-MOD'", "'ARGM-NEG'", "'ARGM-PRD'", "'ARGM-PRP'", "'ARGM-PRR'", "'ARGM-REC'", "'ARGM-TMP'", "'C-ARG0'", "'C-ARG1'", "'C-ARG1-DSP'", "'C-ARG2'", "'C-ARG3'", "'C-ARG4'", "'C-ARGM-ADV'", "'C-ARGM-COM'", "'C-ARGM-CXN'", "'C-ARGM-DIR'", "'C-ARGM-EXT'", "'C-ARGM-GOL'", "'C-ARGM-LOC'", "'C-ARGM-MNR'", "'C-ARGM-PRP'", "'C-ARGM-PRR'", "'C-ARGM-TMP'", "'R-ARG0'", "'R-ARG1'", "'R-ARG2'", "'R-ARG3'", "'R-ARG4'", "'R-ARGM-ADJ'", "'R-ARGM-ADV'", "'R-ARGM-CAU'", "'R-ARGM-COM'", "'R-ARGM-DIR'", "'R-ARGM-GOL'", "'R-ARGM-LOC'", "'R-ARGM-MNR'", "'R-ARGM-TMP'", "'_'"]
    df = pd.read_csv(file)
    all_predictions = [item.strip("[]").split(", ") for item in df['pred_restored']]
    all_gold = [item.strip("[]").split(", ") for item in df['gold_restored']]


    all_predictions = [label.strip() for a_list in all_predictions for label in a_list]
    all_gold = [label.strip() for a_list in all_gold for label in a_list]
    report = classification_report(all_gold, all_predictions, labels=eval_labels_list)
    print(report)

    # I tried to show a beautiful confusion matrix heatmap, but no luck :(

    # cm=confusion_matrix(all_gold, all_predictions, labels=eval_labels_list)
    # df_cm = pd.DataFrame(cm, index = [l[1:-1] for l in eval_labels_list], columns = [l[1:-1] for l in eval_labels_list])
    # sns.set_theme(rc={'figure.figsize':(20, 18)})
    # plot = sns.heatmap(df_cm, annot=True)
    # fig = plot.figure
    # fig.savefig(f'{"/".join(file.split("/")[:-1])}/conf_matrix.png')

    


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
    values_list = [value for value in final_dicts_ids_and_preds.values()]
    return values_list
