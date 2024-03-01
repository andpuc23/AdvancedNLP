# here we (will) have both reading of conllu and features extraction
import pandas as pd

def _get_predicates_from_sentence(lines):
    # here we expect that sentence is a number of lines
    pred_positions = []
    pred_columns = []
    
    for line in lines[2:]:
        split_line = line.split('\t')
        if any([c == 'V' for c in line.split('\t')[11:]]):
            pred_index = split_line.index('V')
            pred_positions.append(int(line.split('\t')[0])-1)
            pred_columns.append(pred_index)
    return pred_positions, pred_columns


    
def process_file(conll_file)->pd.DataFrame:
    big_df = pd.DataFrame(columns=['sentence', 'predicate', 'pred columns', 'pred_columns_values'])
    with open(conll_file) as f:
        text = f.read()
    sentences = text.split('\n\n')  # split by empty line - sent id+text+table with features
    for s in sentences:
        lines = s.split('\n')
        if lines[0].startswith('# propbank'):
            lines = lines[1:]
        if lines[0].startswith('# newdoc'):
            lines = lines[1:]
        if len(lines) > 1:
            sentence_text = lines[1][len('# text = '):]  # everything after "# text = ""
            pred_idxs, pred_cols = _get_predicates_from_sentence(lines)
            
            labels = find_tokens_args(lines, pred_cols)
            
            for idx, col, label in zip(pred_idxs, pred_cols, labels):
                word = lines[idx+2].split('\t')[1]
                big_df.loc[len(big_df.index)] = [sentence_text, word, col, ', '.join(label)]
                # big_df.loc[len(big_df.index)] = [sentence_text, word, idx, col]
        
    print('process_file(): dataframe len:', len(big_df))
    return big_df

def find_tokens_args(lines, pred_cols):
    labels = []
    for i, predicate_col in enumerate(pred_cols):
        labels.append([])
        for line in lines[2:]:
            tags = line.split('\t')
            labels[i].append(tags[predicate_col])
    return labels

    # for sentence, predicate_columns in zip(sent_list, pred_cols):
    #     print(predicate_columns)
    #     if not isinstance(predicate_columns, list):
    #         predicate_columns = [predicate_columns]
    #     for column_index in predicate_columns: #[11, 12]
    #         args = []
    #         tokens_of_args = []
    #         final_list = []
    #         token_with_args = []
    #         for s in sentence:
    #             words = s.split('\t')
    #             # print(new_words)
    #             if not words[0].startswith('#'):
    #                 args.append(words[column_index-1])
    #                 tokens_of_args.append(words[1])
    #                 token_with_args = list(zip(args, tokens_of_args))
    #         for c in token_with_args:
    #             if c[0] not in ('_', 'V'):
    #                 final_list.append(c)
    #     print(final_list)


# def process_file_args(conll_file):
#     big_df = pd.DataFrame(columns=['sentence', 'predicate', 'predicate index', 'pred_columns', 'pred_columns_values'])
#     with open(conll_file) as f:
#         text = f.read()
#     sentences = text.split('\n\n')  # split by empty line - sent id+text+table with features
#     for sentence in sentences:
#         col_numbers = []
#         sent_list = []
#         lines = sentence.split('\n')
#         has_several_lines = len(lines) > 1
#         if has_several_lines and (lines[0].startswith('# newdoc id') or lines[0].startswith('# propbank')):
#             lines = lines[1:]
#             sent_list.append(lines)
#             # print(sent_list)
#             sentence_text = lines[1][len('# text = '):]  # everything after "# text = ""
#             pred_idxs, pred_cols = _get_predicates_from_sentence(lines)
#             col_numbers.append(pred_cols)
#             lll = find_tokens_args(sent_list, col_numbers)
            
#         if has_several_lines:
#             for idx, col in zip(pred_idxs, pred_cols):
#                 #print(lines[idx+2])
#                 word = lines[idx+2].split('\t')[1]
#                 big_df.loc[len(big_df.index)] = [sentence_text, word, idx, col]

#     #col_numbers also has sentences without preds!
#     # sent_id = weblog-blogspot.com_aggressivevoicedaily_20060814163400_ENG_20060814_163400-0004
#     # text = TEHRAN (AFP) -
#     #1	TEHRAN	TEHRAN	PROPN	NNP	Number=Sing	0	root	0:root	_	_	
#     #2	(	(	PUNCT	-LRB-	_	1	punct	1:punct	SpaceAfter=No	_	
#     #3	AFP	AFP	PROPN	NNP	Number=Sing	1	parataxis	1:parataxis	SpaceAfter=No	_	
#     #4	)	)	PUNCT	-RRB-	_	1	punct	1:punct	_	_	
#     #5	-	-	PUNCT	:	_	1	punct	1:punct	_	_	


            
#     print('process_file(): dataframe len:', len(big_df))
    
#     return big_df, col_numbers
# output_1, col_numbers = process_file_args('/Users/sezentuvay/Documents/adnlp/AdvancedNLP/Task_3/data/raw/en_ewt-up-dev.conllu')
        

def extract_features(dataframe)->pd.DataFrame:
    df = pd.DataFrame(columns=['sentences', 'labels'])
    df.sentences = [a + '[SEP]' + b for a, b in zip(dataframe['sentence'].values, dataframe['predicate'].values)]
    # for now I put here the index of word, but it should be the label we predict
    df.labels = dataframe['predicate index']
    return df

