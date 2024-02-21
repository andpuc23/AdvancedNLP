import pandas as pd
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
inputfile = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/output/tryout_dev.csv'
conll_file = pd.read_csv(inputfile, delimiter=',', header=None,  skipinitialspace = False, on_bad_lines='skip')
df = pd.DataFrame(conll_file)
token_list = df[2].tolist()
enhanced_dependencies = df[9].tolist()
gold_list = df[12].tolist()
token_string = ' '.join(map(str, token_list))
nlp.max_length = 5146276
doc = nlp(token_string)
data = []
big_pos_until_target = []

# passive_voice_patterns = [{'E_DEP': 'nsubj:pass'}, {'E_DEP': 'aux:pass'}]
# active_voice_patterns = [{'E_DEP': 'nsubj'}]
#pattern_potential_arg =  #this is for a different feature then pos_until_target!!!
            #subject-verb-object: [{dependency: nsubj or nsubjpass, head: VERB}, {dependency: obj, head:VERB}]
            #subject-verb-compliment: [{dependency: nsubj or nsubjpass, head: VERB}, {dependency: xcomp, ccomp, acomp, or advcl, head: VERB}]
            #verb with prepositional phrase: [{dependency: nsubj or nsubjpass, head: VERB}, case (preposition) to token which is then again attasched to verb with nmod or obl to head: VERB}]
            #Averbial Modifier: [{dependency: advmod, head: VERB}]
            #Verb with Clauses: {dependency: xcomp, ccomp, acomp, or advcl, head: VERB}
    
for i, tok in enumerate(doc):
    token = tok.text 
    pos = tok.pos_
    lemma = tok.lemma_
    dependency = tok.dep_
    head = tok.head
    pos_head = head.pos_
    #dependent = [t.text for t in tok.children]
    morph = tok.morph
    #constituent = [t.text for t in tok.subtree]
    #pos_tags until possible predicate

    '''
    pos_until_target = [] 
    
    while tok != head: #and head comes after tok:
        pos_until_target.append(tok.pos_)
        tok = head
    pos_until_target.append(pos_head)
    '''

    #distance until possible predicate
    # distance = len(pos_until_target)
    # print(distance)


    feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'Dependency': dependency, 'Head': head, 'Head_POS': pos_head,
                'Morphological Feature': morph}#, 'Desc dep': desc_dep}
    data.append(feature_dict)

featured_dict = {'E_DEP': enhanced_dependencies}
data.append(feature_dict)

     
    # voice = None
    # if i < len(enhanced_dependencies):
    #     if enhanced_dependencies[i] in ['nsubj:pass','aux:pass']:
    #         voice = 'Passive'
    #     elif enhanced_dependencies[i] == 'nsubj':
    #         voice = 'Active'

    


df = pd.DataFrame(data=data)
#df['Gold'] = pd.Series(gold_list)    
outputfile = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/output_extract_features/tryout-dev.csv'            
df.to_csv(outputfile,sep='\t', index = False) 
