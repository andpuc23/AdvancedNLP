import pandas as pd
import spacy
import torch

nlp = spacy.load('en_core_web_sm')

def split_gold_column(gold_list):
    predicates = []
    arguments = []
    for element in gold_list:
      if element == '_':
        predicates.append('_')
        arguments.append('_')
      elif element == 'V':
        predicates.append('V')
        arguments.append('_')
      else:
        predicates.append('_')
        arguments.append(element)
    return predicates, arguments

def extract_features(inputfile, outputfile):
    '''
    This function gives the features on a token-level. It takes as input the inputfile, and outputs a 
    csv-file with each feature as a column, The token, pos, lemma, dependency, head, head pos-tag, 
    morphological feature, word embeddings, named entities, path to the predicate in tokens, path to the 
    predicate in pos-tags, context, and the context in pos are all features used for this SRL model. 
    '''
    df = pd.read_csv(inputfile, delimiter=',', header=None,  skipinitialspace = False, on_bad_lines='skip')
    index_within_sent = df[1].tolist()
    token_list = df[2].tolist()
    pos_list = df[4].tolist()
    head_index = df[7].tolist()
    enhanced_dependencies = df[8].tolist()
    gold_list = df[11].tolist()
    predicates, arguments = split_gold_column(gold_list)
    labels_list = df[12].tolist()
    token_string = ' '.join(map(str, token_list))
    nlp.max_length = 5146276
    doc = nlp(token_string)
    assert len(token_list) == len(labels_list)
    data = []
    head_list = []
    dependents_list = []
    for i, tok in enumerate(doc):
        cur = tok
        context = []
        if i > 1:
          prev = doc[i-1]
        else:
          prev = None
        if i < len(doc) - 1:
          next = doc[i+1]
        else:
          next = None
        context.append([prev, cur, next])
        token = tok.text 
        pos = tok.pos_
        cur_pos = pos
        context_pos = []
        if i > 1:
          prev_pos = doc[i-1]
        else:
          prev_pos = None
        if i < len(doc) - 1:
          next_pos = doc[i+1]
        else:
          next_pos = None
        context_pos.append([prev_pos, cur_pos, next_pos])
        lemma = tok.lemma_
        dependency = tok.dep_
        head = tok.head
        head_list.append(str(head))
        pos_head = head.pos_
        dependent = [t.text for t in tok.children]
        dependents_list.append(dependent)
        morph = tok.morph
        named_entities = tok.ent_type_ if tok.ent_type_ else '_'
        word_embedding = torch.tensor(tok.vector)
        path_to_head_text = []
        path_to_head_pos  = []
        try:
            token_label = labels_list[token_list.index(token)]
        except ValueError: # counter not in tokens, there is counter-attack
            token_label = '_'
        label = None
        while cur.has_head() and cur.head != cur and label != 'V':
            path_to_head_text.append(cur.text)
            path_to_head_pos.append(cur.pos_)
            try:
                index = token_list.index(cur.text)
                label = labels_list[index]
                role = enhanced_dependencies[index]
                if label == 'V' or role[:4].lower() == 'root':
                    break
            except:
                break
            cur = cur.head

        feature_dict = {'Token': token, 
                        'PoS': pos, 
                        'Lemma': lemma, 
                        'Dependency': dependency, 
                        'Head': head, 
                        'Head_POS': pos_head,
                        'Morphological Feature': morph, 
                        'Word Embeddings': word_embedding,
                        'Named Entities': named_entities,
                        'Path to head texts':path_to_head_text, 
                        'Path to head POS': path_to_head_pos,
                        'Context': context,
                        'POS Context': context_pos,
                        'Label': token_label}
        data.append(feature_dict)
            
    df = pd.DataFrame(data=data)
    df['Gold'] = pd.Series(gold_list)    
                
    df.to_csv(outputfile, sep='\t', index = False) 