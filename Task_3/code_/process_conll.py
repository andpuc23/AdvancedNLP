# here we (will) have both reading of conllu and features extraction
import pandas as pd

def _get_predicates_from_sentence(lines):
    # here we expect that sentence is a number of lines
    pred_positions = []
    for line in lines[2:]:
        if any([c == 'V' for c in line.split('\t')[11:]]):
            pred_positions.append(int(line.split('\t')[0])-1)
    return pred_positions



def process_file(conll_file)->pd.DataFrame:
    big_df = pd.DataFrame(columns=['sentence', 'predicate', 'predicate index'])
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
        
            pred_idxs = _get_predicates_from_sentence(lines)
            for idx in pred_idxs:
                word = lines[idx+2].split('\t')[1]
                big_df.loc[len(big_df.index)] = [sentence_text, word, idx]

    print('process_file(): dataframe len:', len(big_df))
    return big_df


def extract_features(dataframe)->pd.DataFrame:
    df = pd.DataFrame(columns=['sentences', 'labels'])
    df.sentences = [a + '[SEP]' + b for a, b in zip(dataframe['sentence'].values, dataframe['predicate'].values)]
    # for now I put here the index of word, but it should be the label we predict
    df.labels = dataframe['predicate index']
    return df