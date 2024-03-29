import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
import transformers
from transformers import AutoTokenizer
import numpy as np
from .evaluation import shrink_predictions, class_report

task = "ner"
model_checkpoint = "distilbert-base-uncased"
labels_list = None
metric = load_metric("seqeval")

batch_size = 64

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
    labels_set.remove('V')
    labels_set.remove('C-V')
    labels_list = sorted(list(labels_set))
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


    def tokenize_and_align_labels_pred(self, examples):
        tokenized_sentences = self.tokenizer(examples["sentence"], truncation=False, is_split_into_words=True)
        tokenized_predicates = self.tokenizer(examples["predicate"], truncation=False, is_split_into_words=False)

        tokenized_inputs = dict()
        for key in tokenized_sentences.keys():
            tokenized_inputs[key] = [v1 + v2[1:] for v1, v2 in zip(tokenized_sentences[key], tokenized_predicates[key])] # type: ignore

        list_of_labels_list = [l.split(', ') for l in examples['labels']]

        labels_out = []
        for i, (sentence, predicate, labels_as_list) in enumerate(zip(examples['sentence'], examples['predicate'], list_of_labels_list)):
            tokenized_sentence = self.tokenizer(sentence, truncation=True, is_split_into_words=True)
            labels = []
            pred_position = sentence.index(predicate)
            for word_id in tokenized_sentence.word_ids():
                try:
                    labels.append(-100 if not word_id else self.labels_list.index(labels_as_list[word_id]))
                except:
                    labels.append(self.labels_list.index('_')) # for specific example with 28 words and 27 labels

            count = tokenized_sentence.word_ids().count(pred_position)
            labels += [self.labels_list.index('_')]*count
            labels.append(-100)
            
            labels_out.append(labels)

        tokenized_inputs['labels'] = labels_out
        return tokenized_inputs

    def tokenize_and_align_labels_context(self, examples):
        tokenized_sentences = self.tokenizer(examples["sentence"], truncation=False, is_split_into_words=True)
        tokenized_context = self.tokenizer(examples["context"], truncation=False, is_split_into_words=True)

        tokenized_inputs = dict()
        for key in tokenized_sentences.keys():
            tokenized_inputs[key] = [v1 + v2[1:] for v1, v2 in zip(tokenized_sentences[key], tokenized_context[key])] # type: ignore

        list_of_labels_list = [l.split(', ') for l in examples['labels']]

        labels_out = []
        for i, (sentence, context, labels_as_list) in enumerate(zip(examples['sentence'], examples['context'], list_of_labels_list)):
            tokenized_sentence = self.tokenizer(sentence, truncation=True, is_split_into_words=True)
            labels = []
            pred_positions = [sentence.index(c) if c!='_' else -1 for c in context]
            word_ids = tokenized_sentence.word_ids()
            for word_id in word_ids:
                try:
                    labels.append(-100 if not word_id else self.labels_list.index(labels_as_list[word_id]))
                except:
                    labels.append(self.labels_list.index('_')) # for specific example with 28 words and 27 labels


            base_count = len(self.tokenizer('_')['input_ids'])-2 # type: ignore
            for pred_position in pred_positions:
                if pred_position == -1:
                    labels += [self.labels_list.index('_')]*base_count
                    continue
                count = word_ids.count(pred_position)
                labels += [self.labels_list.index('_')]*count
                
            
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



def predict(trainer, tokenizer, dataset, tokenized_dataset, test_set, ):
    if test_set not in ['train', 'test', 'validation']:
        raise ValueError('Unknown partition of dataset!')
    ds_test = dataset[test_set]
    tds_test = tokenized_dataset[test_set]
    
    predictions_raw, labels, _ = trainer.predict(tds_test)
    predictions = np.argmax(predictions_raw, axis=2)

    list_predictions = [
        [tokenizer.labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tokenizer.labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    val_word_ids = []
    for sentence in tds_test['sentence']:
        val_word_ids.append(tokenizer.tokenizer(sentence, truncation=True, is_split_into_words=True).word_ids())

    df = pd.DataFrame(columns=['sentence', 'prediction', 'gold', 'word_ids'])
    for tokens, prediction, gold, word_ids in zip(tds_test['input_ids'], list_predictions, true_labels, val_word_ids):
        sentence = tokenizer.tokenizer.decode(tokens)
        df.loc[len(df.index)] = [sentence, prediction, gold, word_ids]

    gold_restored = []
    pred_restored = []
    for i, row in df.iterrows():
        sentence = row[0]
        orig_sentence = sentence.split('[SEP]')[0].split(' ')[1:]
        prediction = row[1]
        gold = row[2]
        word_ids = row[3][1:-1]
        gold_restored.append(shrink_predictions(word_ids, gold))
        pred_restored.append(shrink_predictions(word_ids, prediction))

    df['gold_restored'] = gold_restored
    df['pred_restored'] = pred_restored
    df.to_csv(f'data/output/{trainer.args.output_dir.split("/")[1]}_{test_set}.csv')

    class_report(f'data/output/{trainer.args.output_dir.split("/")[1]}_{test_set}.csv')