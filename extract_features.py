import pandas as pd
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')
inputfile = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/output/en_ewt-up-dev.csv'
conll_file = pd.read_csv(inputfile, delimiter=',', header=None,  skipinitialspace = False, on_bad_lines='skip')
df = pd.DataFrame(conll_file)
token_list = df[2].tolist()
enhanced_dependencies = df[9].tolist()
gold_list = df[12].tolist()
token_string = ' '.join(map(str, token_list))
nlp.max_length = 5146276
doc = nlp(token_string)
data = []


passive_voice_patterns = [{'E_DEP': 'nsubj:pass'}, {'E_DEP': 'aux:pass'}]
active_voice_patterns = [{'E_DEP': 'nsubj'}]

#HELP WITH THIS PART!!
matcher = spacy.matcher.Matcher(nlp.vocab)
matcher.add('passive', passive_voice_patterns)
matcher.add('active', active_voice_patterns)

for tok in doc:
    token = tok.text 
    pos = tok.pos_
    lemma = tok.lemma_
    dependency = tok.dep_
    head = tok.head
    pos_head = head.pos_
    dependent = [t.text for t in tok.children]
    morph = tok.morph
    constituent = [t.text for t in tok.subtree]
    matches = matcher(tok)

    feature_dict = {'Token': token, 'PoS': pos, 'Lemma': lemma, 'Dependency': dependency, 'Head': head, 'Head POS': pos_head,
    'Dependent': dependent, 'E_DEP': enhanced_dependencies, 'Morphological Feature': morph, 'Constituent': constituent}#, 'Desc dep': desc_dep}
    data.append(feature_dict)


    #df = pd.DataFrame(data=data)
    #df['Gold'] = pd.Series(gold_list)    
    #outputfile = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/output_extract_features/en_ewt-up-dev.csv'            
    #df.to_csv(outputfile,sep='\t', index = False) 
