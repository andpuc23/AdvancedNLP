import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
import transformers
from transformers import AutoTokenizer
import numpy as np

task = "ner"
model_checkpoint = "distilbert-base-uncased"
labels_list = None
metric = metric = load_metric("seqeval")

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
    return list(set(ds['train'].features['label'].names))


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
    
    def tokenize(self, input):
        return map(self._tokenize_input_string, input)
    
    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples['tokens'])

        labels = []
        for i, label in enumerate(self.labels_list):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
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

# preprocess
#   tokenizer
#   tokenize + align
#   do that with datasets

# training
#   collate
#   trainer
#   train

