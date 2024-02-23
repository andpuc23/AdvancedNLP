from argparse import ArgumentParser
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

from conllu_to_df import process_file
from extract_features import extract_features
import numpy as np


def extract_features_and_labels(trainingfile):
    features = []
    gold = []
    with open(trainingfile, 'r', encoding='utf-8') as infile:
        for line in infile:
            components = line.rstrip('\n').split('\t')
            if len(components) > 0:
                token = components[0]
                pos = components[1]
                lemma = components[2]
                dependency = components[3]
                head = components[4]
                head_pos = components[5]
                morph_feat = components[6]
                ent_type = components[7]
                ent_dep = components[8]
                voice = components[9]
                gold_a = components[-1]
                feature_dict = {'Token': token,
                                'PoS': pos,
                                'Lemma': lemma, 
                                'dependency': dependency, 
                                'head': head, 
                                'head pos': head_pos,
                                'morph feature': morph_feat,
                                'entity type': ent_type,
                                'entity dep': ent_dep,
                                'voice': voice}
                features.append(feature_dict)
                gold.append(gold_a)
    return features, gold


if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('-T', '--train',
						dest='train', help='Name of the train file',
						default='en_ewt-up-train.conllu', type=str)
	parser.add_argument('-S', '--test',
						dest='test', help='Name of the test file',
						default='en_ewt-up-test.conllu', type=str)
	parser.add_argument('-D', '--dev',
						dest='dev', help='Name of the dev file',
						default='en_ewt-up-dev.conllu', type=str)
	parser.add_argument('-F', '--file',
						dest='file', help='The input file directory',
						default='../UP-1.0/input/', type=str)
	parser.add_argument('-O', '--output',
						dest='output', help='Name of output directory',
						default='../UP-1.0/output/', type=str)
	parser.add_argument('-Q', '--feature',
						dest='feature', help='Name of feature directory',
						default='../UP-1.0/feature/', type=str)
	args = parser.parse_args()

	train_file = args.file + args.train
	test_file = args.file + args.test
	dev_file = args.file + args.dev

	out_train_file = args.output + args.train
	out_test_file = args.output + args.test
	out_dev_file = args.output + args.dev

	feat_train_file = args.feature + args.train
	feat_test_file = args.feature + args.test
	feat_dev_file = args.feature + args.dev

	extract_features(out_dev_file, feat_dev_file)
	extract_features(out_train_file, feat_train_file)
	extract_features(out_test_file, feat_test_file)

	vectorizer = DictVectorizer()

	train_df = pd.read_csv(feat_train_file, sep='\t', low_memory=False)
	train_data = train_df.to_dict('records')
	train_vectorized = vectorizer.fit_transform(train_data)

	test_df = pd.read_csv(feat_test_file, sep='\t', low_memory=False)
	test_data = test_df.to_dict('records')
	test_vectorized = vectorizer.fit_transform(test_data)

	dev_df = pd.read_csv(feat_dev_file, sep='\t', low_memory=False)
	dev_data = dev_df.to_dict('records')
	dev_vectorized = vectorizer.fit_transform(dev_data)

	train_vectorized = np.array(train_vectorized)
	test_vectorized = np.array(test_vectorized)
	dev_vectorized = np.array(dev_vectorized)

	X = train_vectorized.data[:, :-1]
	X_dev = dev_vectorized.data[:, :-1]

	y = train_vectorized[:, -1]
	y_dev = dev_vectorized[:, -1]

	model = LogisticRegression().fit(X, y)
	y_pred = model.predict(X_dev)

	print("METRICS")
	print('precision', precision_score(y_dev, y_pred), sep='\t')
	print('recall', recall_score(y_dev, y_pred), sep='\t')
	print('f1', f1_score(y_dev, y_pred), sep='\t')
