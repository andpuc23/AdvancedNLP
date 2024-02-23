import pandas as pd


def process_file(conll_file, output_file, save_file=True):
	'''
	Takes a conllu file, removes comments and sentence separating
	lines, duplicates sentences with multiple predicates.
	'''

	with open(conll_file, 'r', encoding='utf8') as f:
		data = f.read()
		sentences = data.strip().split('\n\n')

	all_df = []

	for sent in sentences:
		rows = sent.strip().split('\n')
		rows = [row for row in rows if not row[0].startswith('#')]
		columns = [row.split('\t') for row in rows]

		if len(columns[0]) > 10:
			df = pd.DataFrame(columns)
			num_predicates = df[10].apply(lambda x: 1 if x != '_' else 0).sum()

			if num_predicates == df.shape[1] - 11:
				for j in range(num_predicates):
					df_new = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10 + j + 1]]
					new_names = {col: k for k, col in enumerate(df_new.columns)}
					df_new = df_new.rename(columns=new_names)
					all_df.append(df_new)

	big_df = pd.concat(all_df, ignore_index=True)
	big_df.columns = ['id', 'token', 'lemma', 'pos', 'xpos', 'morphology', 'head_id', 'deprel', 'deps', 'misc', 'propbank', 'srl']
	big_df['sentence'] = (df.id == 1).cumsum()
	big_df.id = big_df.sentence.astype(str) + '.' + big_df.id.astype(str)
	big_df.head_id = big_df.sentence.astype(str) + '.' + big_df.head_id.astype(str)
	big_df.drop('sentence', axis=1, inplace=True)

	if save_file:
		big_df.to_csv(output_csv, index=False, header=False)

	return big_df


def add_head_pos_tag(conll_df):
	'''
	conll_df must be a df as returned by process_file
	'''
	conll_df['head_pos'] = df['head_id'].apply(lambda x: id_pos.get(x, '_'))
	return conll_df


if __name__ == '__main__':
	train_file = '../UP-1.0/input/en_ewt-up-train.conllu'

	df = process_file(train_file, None, save_file=False)
	df = add_head_pos_tag(df)
