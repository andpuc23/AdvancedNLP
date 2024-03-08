import pandas as pd

def _get_predicates_from_sentence(lines):
    '''
    
    '''
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



def process_file(conll_file:str)->pd.DataFrame:
    '''
    here we process .conllu file into dataframe and extract predicate from it
    @param conll_file filename of the file with data
    @return dataframe with columns 'sentence', 'predicate', 'pred columns', 'labels'
    '''

    big_df = pd.DataFrame(columns=['sentence', 'predicate', 'pred columns', 'labels'])
    with open(conll_file) as f:
        text = f.read()
    sentences = text.split('\n\n')  # split by empty line - sent id+text+table with features
    # remove propbank name and newdoc id - we don't need them here
    for s in sentences:
        lines = s.split('\n')
        if lines[0].startswith('# propbank'):
            lines = lines[1:]
        if lines[0].startswith('# newdoc'):
            lines = lines[1:]


        if len(lines) > 1:
            sentence_words_list = [l.split('\t')[1] for l in lines[2:]]  # skip sentence id and text
            pred_idxs, pred_cols = _get_predicates_from_sentence(lines)  # extract predicate index in the sentence

            labels = find_tokens_args(lines, pred_cols)

            for idx, col, label in zip(pred_idxs, pred_cols, labels):
                word = lines[idx+2].split('\t')[1]
                big_df.loc[len(big_df.index)] = [sentence_words_list, word, col, ', '.join(label)]

    print('process_file(): dataframe len:', len(big_df))
    return big_df


def _get_context_of_predicate(sentence_words_list, word, idx):
    context = ['_', word, '_']

    if idx >= 1:
        context[0] = sentence_words_list[idx-1]

    if idx < len(sentence_words_list)-1:
        context[2] = sentence_words_list[idx+1]

    return context


def advanced_process_file(conll_file)->pd.DataFrame:
    big_df = pd.DataFrame(columns=['sentence', 'predicate', 'pred columns', 'context', 'labels'])
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
                context = _get_context_of_predicate(sentence_words_list, word, idx)
                big_df.loc[len(big_df.index)] = [sentence_words_list, word, col, context, ', '.join(label)]

    print('advanced_process_file(): dataframe len:', len(big_df))
    return big_df


def find_tokens_args(lines, pred_cols):
    labels = []
    for i, predicate_col in enumerate(pred_cols):
        labels.append([])
        for line in lines[2:]:
            tags = line.split('\t')
            try:
                label = tags[predicate_col]
                if label == '':
                    label = '_'
                labels[i].append(label)
            except:
                pass
    return labels
