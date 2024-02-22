import pandas as pd
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
# inputfile = '../UP-1.0/output/dev.csv'

def extract_features(inputfile, outputfile):

    # inputfile = 'C:/Users/snipercapt/Desktop/ANLP/AdvancedNLP/Task_2/UP-1.0/output/dev.csv'
    conll_file = pd.read_csv(inputfile, delimiter=',', header=None,  skipinitialspace = False, on_bad_lines='skip')
    df = pd.DataFrame(conll_file)
    index_within_sent = df[1].tolist()
    token_list = df[2].tolist()
    pos_list = df[4].tolist()

    enhanced_dependencies = df[8].tolist()
    gold_list = df[12].tolist()
    token_string = ' '.join(map(str, token_list))
    nlp.max_length = 5146276
    doc = nlp(token_string)
    data = []
    # passive_voice_patterns = [{'E_DEP': 'nsubj:pass'}, {'E_DEP': 'aux:pass'}]
    # active_voice_patterns = [{'E_DEP': 'nsubj'}]
    #pattern_potential_arg =  #this is for a different feature then pos_until_target!!!
                #subject-verb-object: [{dependency: nsubj or nsubjpass, head: VERB}, {dependency: obj, head:VERB}]
                #subject-verb-compliment: [{dependency: nsubj or nsubjpass, head: VERB}, {dependency: xcomp, ccomp, acomp, or advcl, head: VERB}]
                #verb with prepositional phrase: [{dependency: nsubj or nsubjpass, head: VERB}, case (preposition) to token which is then again attasched to verb with nmod or obl to head: VERB}]
                #Averbial Modifier: [{dependency: advmod, head: VERB}]
                #Verb with Clauses: {dependency: xcomp, ccomp, acomp, or advcl, head: VERB}
    head_list = []
    for tok in doc:
        token = tok.text 
        pos = tok.pos_
        lemma = tok.lemma_
        dependency = tok.dep_
        head = tok.head
        head_list.append(str(head))
        pos_head = head.pos_
        #dependent = [t.text for t in tok.children]
        morph = tok.morph
        #constituent = [t.text for t in tok.subtree]
        #pos_tags until possible predicate
        '''
        pos_until_target = [] 
        while tok != head: #and head[] comes after tok:
            pos_until_target.append(tok.pos_)
            tok = head
        pos_until_target.append(pos_head)
        '''

        
        feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'Dependency': dependency, 'Head': head, 'Head_POS': pos_head,
                    'Morphological Feature': morph}#, 'Desc dep': desc_dep}
        data.append(feature_dict)
    
    #voice
    voice = []
    passive_set = {'nsubj:pass', 'aux:pass'}
    for element in enhanced_dependencies:
        if element in passive_set:
            voice.append('passive')
        elif 'nsubj' in element:
            voice.append('active')
        else:
            voice.append(None)
    
    #pos until target:
    #does not work yet
    big_pos_until_target = []
    tuples_list = list(zip(index_within_sent, token_list, pos_list, head_list))
    for i, j, element in enumerate(tuples_list):
        pos_until_target = []
        if element[1] == element[3]:
            possible_pred_index = element[0]
        if int(possible_pred_index) < int(element[0]):
            pos_until_target.append(element[2])
        elif element[1] == element[3]:
            big_pos_until_target.append(pos_until_target)

    #distance until possible predicate
    # distance = len(pos_until_target)

    #position of predicate in the sentence
    #
    featured_dict_1 = {'E_DEP': enhanced_dependencies, 'voice': voice}
    feature_dict.update(featured_dict_1)
    data.append(feature_dict)
        
    df = pd.DataFrame(data=data)
    #df['Gold'] = pd.Series(gold_list)    
    #             
    # outputfile = 'C:/Users/snipercapt/Desktop/ANLP/AdvancedNLP/Task_2/UP-1.0/features/dev.csv'
    df.to_csv(outputfile, sep='\t', index = False) 

inputfile = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/output/tryout_dev.csv'
outputfile = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/output_extract_features/tryout-dev.csv'
extract_features(inputfile, outputfile)
