# here we (will) have both reading of conllu and features extraction
import pandas as pd

def _get_predicates_from_sentence(lines):
    # here we expect that sentence is a number of lines
    pred_positions = []
    pred_columns = []
    
    for line in lines[2:]:
        split_line = line.split('\t')
        if any([c == 'V' for c in split_line[11:]]):
            pred_index = split_line.index('V')
            if not split_line[0].isdigit():
                print('found shit:', split_line[0:1])
            else:
                pred_positions.append(int(split_line[0])-1)
                pred_columns.append(pred_index)
    return pred_positions, pred_columns


    
def process_file(conll_file)->pd.DataFrame:
    big_df = pd.DataFrame(columns=['sentence', 'predicate', 'pred columns', 'labels'])
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
            sentence_words_list = [l.split('\t')[1] for l in lines[2:]]
            pred_idxs, pred_cols = _get_predicates_from_sentence(lines)
            
            labels = find_tokens_args(lines, pred_cols)
            
            for idx, col, label in zip(pred_idxs, pred_cols, labels):
                word = lines[idx+2].split('\t')[1]
                big_df.loc[len(big_df.index)] = [sentence_words_list, word, col, ', '.join(label)]
        
    print('process_file(): dataframe len:', len(big_df))
    return big_df

def find_tokens_args(lines, pred_cols):
    labels = []
    for i, predicate_col in enumerate(pred_cols):
        labels.append([])
        for line in lines[2:]:
            tags = line.split('\t')
            try:
                labels[i].append(tags[predicate_col])
            except:
                pass
    return labels
    

def extract_features(dataframe)->pd.DataFrame:
    raise ValueError("do not use this method!!")
    df = pd.DataFrame(columns=['sentences', 'labels', 'labels_list'])
    df.sentences = [a + ['[SEP]', b] for a, b in zip(dataframe['sentence'].values, dataframe['predicate'].values)]
    # for now I put here the index of word, but it should be the label we predict
    df.labels = dataframe['pred columns values']
    df.labels_list = [l.split(', ') for l in df.labels]
    return df

