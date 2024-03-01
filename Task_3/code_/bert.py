import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
import transformers
from transformers import AutoTokenizer
import numpy as np

task = "ner"
model_checkpoint = "distilbert-base-uncased"
labels_list = None
metric = load_metric("seqeval")

batch_size = 16 # subject to change, the bigger the better, but should fit into memory

def convert_to_dataset(train:pd.DataFrame, 
                       val:pd.DataFrame, 
                       test:pd.DataFrame)->DatasetDict:
    global labels_list
    train_ds = Dataset.from_pandas(train)
    val_ds = Dataset.from_pandas(val)
    test_ds = Dataset.from_pandas(test)

    ds = DatasetDict()

    ds['train'] = train_ds
    ds['validation'] = val_ds
    ds['test'] = test_ds

    if not labels_list:
        labels_list = get_labels_list_from_dataset(ds)

    return ds


def get_labels_list_from_dataset(ds:DatasetDict):
    labels_set = set()
    for label in ds['train']['labels']:
        vals = label.split(', ')
        for v in vals:
            labels_set.add(v)
    if '' in labels_set:
        labels_set.remove('')
    labels_list = list(labels_set)
    return labels_list

class Tokenizer:
    def __init__(self, model_checkpoint, labels_list) -> None:
        """
        :param model_checkpoint
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast), "tokenizer is not PreTrainedTokenizerFast!"

        self.labels_list = labels_list


    def _tokenize_input_string(self, input):
        if isinstance(input, str):
            return self.tokenizer(input, truncation=True)
        elif isinstance(input, list):
            return self.tokenizer(input, truncation=True, is_split_into_words=True)
        else:
            raise TypeError(f'tokenizer input should be str or list, got {type(input)}')

    
    def tokenize_and_align_labels(self, examples):
        labels_data = []
        for s, l in zip(examples['sentences'], examples['labels_list']):
            l = l.split(', ')
            tokenized_inputs = self.tokenizer(s, truncation=True)
            labels = []
            for word_id in tokenized_inputs.word_ids():
                if word_id == None:
                    labels.append(-100)
                else:
                    try:
                        labels.append(l[word_id])
                    except IndexError:
                        labels.append(self.labels_list.index('V'))
            labels_data.append(labels)

        tokenized_inputs["labels"] = labels_data
        return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class DataCollator:
    pass

# preprocess
#   tokenizer
#   tokenize + align
#   do that with datasets

# training
#   collate
#   trainer
#   train

