import pandas as pd
conll_file = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/input/en_ewt-up-dev.conllu'


df = pd.DataFrame()
sentences = []
all_df = []
with open(conll_file, 'r', encoding="utf8") as file:
  data = file.read()
  sentences = data.strip().split('\n\n')

all_df = []
for sentence in sentences:
   rows = sentence.strip().split('\n')
   rows = [row for row in rows if not row[0].startswith('#')]
   columns = [row.split('\t') for row in rows]
   if len(columns[0])>10:
    df = pd.DataFrame(columns)  
    num_predicates = df[10].apply(lambda x: 1 if x !='_' else 0).sum()

    if num_predicates == df.shape[1]-11:
        for j in range(num_predicates):
            df_new = df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,10+j+1]]
            new_names = {col: k for k, col in enumerate(df_new.columns)}
            df_new = df_new.rename(columns=new_names)
            all_df.append(df_new)

big_df = pd.concat(all_df, ignore_index=True)
print(len(big_df))
output_csv = '/Users/sezentuvay/Documents/advanced_nlp/Assignment2AdvancedNLP/data/output/en_ewt-up-dev.csv'
big_df.to_csv(output_csv)
