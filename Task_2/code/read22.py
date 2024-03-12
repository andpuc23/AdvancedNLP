from re import I
import pandas as pd
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
inputfile = '../UP-1.0/output/dev.csv'

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
    conll_file = pd.read_csv(inputfile, delimiter=',', header=None,  skipinitialspace = False, on_bad_lines='skip')
    df = pd.DataFrame(conll_file)
    sentences = []
    sentence = ""
    for index, row in df.iterrows():
        if str(row[1]) == '1':
            sentences.append('\n\n')
            sentences.append(row)
    #print(sentences)
    index_within_sent= df[0].tolist()
    token_list = df[1].tolist()
    pos_list = df[3].tolist()
    head_index = df[6].tolist()
    enhanced_dependencies = df[7].tolist()
    gold_list = df[11].tolist()
    predicates, arguments = split_gold_column(gold_list)
    print(predicates, arguments)
    token_string = ' '.join(map(str, token_list))
    nlp.max_length = 5146276
    doc = nlp(token_string)
    data = []
    head_list = []
    dependents_list = []
    for tok in doc:
        token = tok.text 
        pos = tok.pos_
        lemma = tok.lemma_
        dependency = tok.dep_
        head = tok.head
        head_list.append(str(head))
        pos_head = head.pos_
        dependent = [t.text for t in tok.children]
        dependents_list.append(dependent)
        morph = tok.morph
        named_entities = tok.ent_type_
        feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'Dependency': dependency, 'Head': head, 'Head_POS': pos_head,
                    'Morphological Feature': morph, 'NE': named_entities}
        data.append(feature_dict)

    #pos of dependents!! does this work properly? I wanted to implement a feature that gets the POS-tags of the tokens until the headword is reached, I could not implement it so that is why I came up with something similar like this:
    # big_pos_tags_to_dependents = []
    # big_distance = []
    # for element in dependents_list:
    #     doc = nlp(str(element))
    #     pos_tags = []
    #     for tok in doc:
    #         pos = tok.pos_
    #         pos_tags.append(pos)
    #     big_pos_tags_to_dependents.append(pos_tags)
    #     big_distance.append(len(pos_tags))

    # feature_dict.update({'Big_distance':big_distance})

    # print('extracted distance to deps')

    # I want to implement a feature that uses these patterns! I think they are very useful for finding the arguments!!!
    #pattern_potential_arg =  
                #subject-verb-object: [{dependency: nsubj or nsubjpass, head: VERB}, {dependency: obj, head:VERB}]
                #subject-verb-compliment: [{dependency: nsubj or nsubjpass, head: VERB}, {dependency: xcomp, ccomp, acomp, or advcl, head: VERB}]
                #verb with prepositional phrase: [{dependency: nsubj or nsubjpass, head: VERB}, case (preposition) to token which is then again attasched to verb with nmod or obl to head: VERB}]
                #Averbial Modifier: [{dependency: advmod, head: VERB}]
                #Verb with Clauses: {dependency: xcomp, ccomp, acomp, or advcl, head: VERB}

    #CODE: Position of predicate in the sentence
    index_within_sent = df[1].tolist()
    lengths = []
    new_lenghts = []
    for i in range(len(index_within_sent)):
        if index_within_sent[i] == 1:
            if i>0:
                lengths.append(index_within_sent[i-1])
    lengths[-1] = lengths[-1]+1
    lengths = lengths[1:]

    result = list(zip(df.loc[df[11] != '_', 11], df.loc[df[11] != '_', 1]))
    #print(result)

    #CODE: Phrase type, take from assignment 1!!!
    #phrase_type = ['']*len(doc)
    patterns = [{'POS': 'VBP', 'OP': '?'},
                {'LEMMA': 'have', 'TAG': 'VBP', 'OP': '?'},
                {'TEXT': 'not', 'OP': '?'},
                {'TAG': 'VBP', 'OP': '?'},
                {'POS': 'VERB', 'OP': '?'},
                {'POS': 'ADV', 'OP': '*'},
                {'POS': 'AUX', 'OP': '*'},
                {'POS': 'VERB', 'OP': '+'}]

    # matcher = spacy.matcher.Matcher(nlp.vocab)
    # matcher.add("Verb phrase", [patterns])
    # # VP
    # matches = matcher(doc)
    # for _, start, end in matches:
    #     for i in range(start, end):
    #         phrase_type[i] = 'VP'
    # # spans = [doc[start:end] for _, start, end in matches]

    # ## PP
    # for i, element in enumerate(doc):
    #     if element.pos_ == 'ADP':
    #         phrase_type[i] = 'PP'
    # # print('PPs:', pps)
    # ## NP
    # for i, element in enumerate(doc):
    #     if element.text in set(doc.noun_chunks):
    #         phrase_type[i] = 'NP'   
    
    # # print('NPs:', [np for np in doc.noun_chunks])
    # print('extracted phrase type')


    featured_dict_1 = {'E_DEP': enhanced_dependencies, 'voice': voice}
    data.append(feature_dict)
        
    df = pd.DataFrame(data=data)
    df['Gold'] = pd.Series(gold_list)    
                
    # outputfile = 'C:/Users/snipercapt/Desktop/ANLP/AdvancedNLP/Task_2/UP-1.0/features/dev.csv'
    df.to_csv(outputfile, sep='\t', index = False) 
#file = 'outje'
inputfile = '/content/drive/MyDrive/task22/outje222.csv'
outputfile = '/content/drive/MyDrive/task22/outje333.csv'
extract_features(inputfile, outputfile)
