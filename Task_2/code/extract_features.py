import pandas as pd
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
# inputfile = '../UP-1.0/output/dev.csv'

def extract_features(inputfile, outputfile):

    # inputfile = 'C:/Users/snipercapt/Desktop/ANLP/AdvancedNLP/Task_2/UP-1.0/output/dev.csv'
    df = pd.read_csv(inputfile, delimiter=',', header=None,  skipinitialspace = False, on_bad_lines='skip')
    index_within_sent = df[1].tolist()
    token_list = df[2].tolist()
    pos_list = df[4].tolist()
    head_index = df[7].tolist()
    enhanced_dependencies = df[8].tolist()
    gold_list = df[11].tolist()
    labels_list = df[12].tolist()
    token_string = ' '.join(map(str, token_list))
    nlp.max_length = 5146276
    doc = nlp(token_string)
    assert len(token_list) == len(labels_list)
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
        
        path_to_head_text = []
        path_to_head_pos  = []
        cur = tok
        try:
            token_label = labels_list[token_list.index(token)]
        except ValueError: # counter not in tokens, ther is counter-attack
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
                        'Path to head texts':path_to_head_text, 
                        'Path to head POS': path_to_head_pos, 
                        'Label': token_label}#, 'Desc dep': desc_dep}
        data.append(feature_dict)
    
    #Voice
    voice = []
    passive_set = {'nsubj:pass', 'aux:pass'}
    for element in enhanced_dependencies:
        if element in passive_set:
            voice.append('passive')
        elif 'nsubj' in element:
            voice.append('active')
        else:
            voice.append(None)

    print('exctacted voice')
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

    # labels
    # for ent in doc.ents:
    #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
    
    # entity
    feature_dict.update({'Entity_type': [tok.ent_type_ for tok in doc]})

    

    # I want to implement a feature that uses these patterns! I think they are very useful for finding the arguments!!!
    #pattern_potential_arg =  
                #subject-verb-object: [{dependency: nsubj or nsubjpass, head: VERB}, {dependency: obj, head:VERB}]
                #subject-verb-compliment: [{dependency: nsubj or nsubjpass, head: VERB}, {dependency: xcomp, ccomp, acomp, or advcl, head: VERB}]
                #verb with prepositional phrase: [{dependency: nsubj or nsubjpass, head: VERB}, case (preposition) to token which is then again attasched to verb with nmod or obl to head: VERB}]
                #Averbial Modifier: [{dependency: advmod, head: VERB}]
                #Verb with Clauses: {dependency: xcomp, ccomp, acomp, or advcl, head: VERB}

    #CODE: Position of predicate in the sentence
    
    #CODE: Phrase type, take from assignment 1!!!
    phrase_type = ['']*len(doc)
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
    feature_dict.update(featured_dict_1)
    data.append(feature_dict)
        
    df = pd.DataFrame(data=data)
    df['Gold'] = pd.Series(gold_list)    
                
    # outputfile = 'C:/Users/snipercapt/Desktop/ANLP/AdvancedNLP/Task_2/UP-1.0/features/dev.csv'
    df.to_csv(outputfile, sep='\t', index = False) 

# inputfile = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/output/tryout_dev.csv'
# outputfile = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/output_extract_features/tryout-dev.csv'
# extract_features(inputfile, outputfile)
