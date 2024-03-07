import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
import transformers
from transformers import AutoTokenizer
import numpy as np

task = "ner"
model_checkpoint = "distilbert-base-uncased"
labels_list = None
metric = load_metric("seqeval")

batch_size = 32 # subject to change, the bigger the better, but should fit into memory

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
    
    for ds_name in ['train', 'test', 'validation']:
        for label in ds[ds_name]['labels']:
            vals = label.split(', ')
            for v in vals:
                labels_set.add(v)
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


    def tokenize_align_labels_no_pred(self, examples):
        global labels_list
        tokenized_sentences = self.tokenizer(examples["sentence"], truncation=True, is_split_into_words=True)

        list_of_labels_list = [l.split(', ') for l in examples['labels']]

        labels_out = []
        for i, (sentence, labels_as_list) in enumerate(zip(examples['sentence'], list_of_labels_list)):
            tokenized_sentence = self.tokenizer(sentence, truncation=True, is_split_into_words=True)
            labels = []
            for word_id in tokenized_sentence.word_ids():
                try:
                    labels.append(-100 if word_id is None else labels_list.index(labels_as_list[word_id]))
                except:
                    labels.append(labels_list.index('_')) # for specific example with 28 words and 27 labels
            labels_out.append(labels)

        tokenized_sentences['labels'] = labels_out
        
        return tokenized_sentences

    
    def tokenize_and_align_labels_pred(self, examples):
        global labels_list
        tokenized_sentences = self.tokenizer(examples["sentence"], truncation=False, is_split_into_words=True)
        tokenized_predicates = self.tokenizer(examples["predicate"], truncation=False, is_split_into_words=False)

        tokenized_inputs = dict()
        for key in tokenized_sentences.keys():
            tokenized_inputs[key] = [v1 + v2[1:] for v1, v2 in zip(tokenized_sentences[key], tokenized_predicates[key])]
        
        list_of_labels_list = [l.split(', ') for l in examples['labels']]

        labels_out = []
        for i, (sentence, predicate, labels_as_list) in enumerate(zip(examples['sentence'], examples['predicate'], list_of_labels_list)):
            # sentence = ex['sentence']
            tokenized_sentence = self.tokenizer(sentence, truncation=True, is_split_into_words=True)
            labels = []
            pred_position = sentence.index(predicate)
            for word_id in tokenized_sentence.word_ids():
                try:
                    labels.append(-100 if word_id is None else labels_list.index(labels_as_list[word_id]))
                except:
                    labels.append(labels_list.index('_')) # for specific example with 28 words and 27 labels
            
            count = tokenized_sentence.word_ids().count(pred_position)

            labels += [labels_list.index('_')]*count
            labels.append(-100)
            
            labels_out.append(labels)

        tokenized_inputs['labels'] = labels_out
        
        return tokenized_inputs

    def tokenize_and_align_labels_context(self, examples):
        global labels_list
        tokenized_sentences = self.tokenizer(examples["sentence"], truncation=False, is_split_into_words=True)
        tokenized_context = self.tokenizer(examples["context"], truncation=False, is_split_into_words=True)

        tokenized_inputs = dict()
        for key in tokenized_sentences.keys():
            tokenized_inputs[key] = [v1 + v2[1:] for v1, v2 in zip(tokenized_sentences[key], tokenized_context[key])]
        
        list_of_labels_list = [l.split(', ') for l in examples['labels']]

        labels_out = []
        for i, (sentence, context, labels_as_list) in enumerate(zip(examples['sentence'], examples['context'], list_of_labels_list)):
            tokenized_sentence = self.tokenizer(sentence, truncation=True, is_split_into_words=True)
            labels = []
            pred_positions = [sentence.index(c) if c!='_' else -1 for c in context]
            word_ids = tokenized_sentence.word_ids()
            for word_id in word_ids:
                try:
                    labels.append(-100 if word_id is None else labels_list.index(labels_as_list[word_id]))
                except:
                    labels.append(labels_list.index('_')) # for specific example with 28 words and 27 labels
            

            base_count = len(self.tokenizer('_')['input_ids'])-2
            for pred_position in pred_positions:
                if pred_position == -1:
                    labels += [labels_list.index('_')]*base_count
                    continue
                count = word_ids.count(pred_position)
                labels += [labels_list.index(labels_as_list[pred_position])]*count
            labels.append(-100)
            
            labels_out.append(labels)

        tokenized_inputs['labels'] = labels_out
        return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [labels_list[p] for (p, l) in zip(prediction, label) if l != -100 and p < len(labels_list)]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_list[l] for (p, l) in zip(prediction, label) if l != -100 and p < len(labels_list)]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
