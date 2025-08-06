# %%
import os
import pandas as pd


# %%
def get_all_files(folder_name):
    # Change the directory
    os.chdir(folder_name)
    # iterate through all file
    file_path_list =[]
    for file in os.listdir():
        print(file)
        file_path = f"{folder_name}/{file}"
        file_path_list.append(file_path)
    return file_path_list

# %% [markdown]
# ### 1. NQ Dataset Analysis

# %%
doc_folder = '/Users/abhilashamangal/Documents/Semantic Search/data/doc-nq910'

# %%
files = get_all_files(doc_folder)

# %%
df_question = pd.read_csv(files[0],sep = '\t') 

# %%
df_question.head()

# %%
len(df_question)

# %%
df_docs = pd.read_csv(files[1],sep = '\t') 

# %%
df_docs.head()

# %%
len(df_docs)

# %%
df_question['answers']

# %%
df_question['answers'].isnull().sum()

# %%


# %%
i =0
for index, row in df_docs.iterrows():
        i=i+1
        print("Processing i",i)
        id_ = row['id']
        text = row['text']
        title = row['title']
        
        print("Token Length---",len(text.split(" ")))
        
        token_array = text.split(" ")
        
        for i in range(len(token_array)):
            start =i
            end = i+512
            context = ""
            if i < end :
                

# %%


# %% [markdown]
# ### 2. LongNQ-doc

# %%
doc_folder_long = '/Users/abhilashamangal/Documents/Semantic Search/data/LongNQ-docs/'

# %%
files_nq = get_all_files(doc_folder_long)

# %%
df_question_long = pd.read_csv(files_nq[0],sep = '\t') 

# %%
df_question_long.head()

# %%
len(df_question_long)

# %%
import numpy as np

# %%
df_question_long['answers'].replace('-', np.nan, inplace=True)

# %%
df_question_long['answers'].isnull().sum()

# %%
df_question_long.dropna(subset=['answers'], inplace=True)

# %%
len(df_question_long)

# %%
df_question_long.head()

# %%


# %%
df_docs_long = pd.read_csv(files_nq[1],sep = '\t') 

# %%
df_docs_long.head()

# %%
len(df_docs_long)

# %% [markdown]
# ### 3. Msmarco

# %%
doc_folder_msmarco = '/Users/abhilashamangal/Documents/Semantic Search/data/msmarco/'

# %%
files_msmarco = get_all_files(doc_folder_msmarco)

# %%
df_corpus = pd.read_json(files_msmarco[2],lines=True)

# %%
len(df_corpus)

# %%
pd.set_option('display.max_colwidth', 255)

# %%
df_corpus.head()

# %%
### Queries 

# %%
df_queries = pd.read_json(files_msmarco[3],lines=True)

# %%
df_queries.head()

# %%
doc_folder_msmarco_t = '/Users/abhilashamangal/Documents/Semantic Search/data/msmarco/qrels'

# %%
files_msmarco_t = get_all_files(doc_folder_msmarco_t)

# %%
df_queries_t = pd.read_csv(files_msmarco_t[0],sep = '\t')

# %%
df_queries_t.head()

# %%
df_queries_t['score'].unique()

# %%
len(df_queries_t)

# %%
df_queries_d = pd.read_csv(files_msmarco_t[1],sep = '\t')

# %%
df_queries_d.head()

# %%
len(df_queries_d)

# %%
df_queries_tt = pd.read_csv(files_msmarco_t[2],sep = '\t')

# %%
df_queries_tt.head()

# %%
len(df_queries_tt)

# %%


# %%


# %%


# %% [markdown]
# ### 4. Trec-covid analysis

# %%
doc_folder_trec = '/Users/abhilashamangal/Documents/Semantic Search/data/trec-covid/'

# %%
files_trec = get_all_files(doc_folder_trec)

# %%
df_corpus_trec = pd.read_json(files_trec[1],lines=True)

# %%
df_corpus_trec.head()

# %%
df_queries_trec = pd.read_json(files_trec[2],lines=True)

# %%
df_queries_trec.head()

# %%
doc_folder_trec_q = '/Users/abhilashamangal/Documents/Semantic Search/data/trec-covid/qrels'

# %%
files_trec_q = get_all_files(doc_folder_trec_q)

# %%
df_queries_trec_t = pd.read_csv(files_trec_q[0],sep = '\t')

# %%
len(df_queries_trec_t)

# %%
df_queries_trec_t.head(30)

# %%
df_queries_trec_t['score'].unique()

# %%
df_queries_trec_t['query-id'].unique()

# %%
doc_folder_hotpotqa = '/Users/abhilashamangal/Documents/Semantic Search/data/hotpotqa'

# %%
files_hotpotqa  = get_all_files(doc_folder_hotpotqa)

# %%
df_corpus_hotpotqa = pd.read_json(files_hotpotqa[1],lines=True)

# %%
df_corpus_hotpotqa.head()

# %%
len(df_corpus_hotpotqa)

# %%
df_queries_hotpotqa = pd.read_json(files_hotpotqa[2],lines=True)

# %%
df_queries_hotpotqa.head()

# %%
len(df_queries_hotpotqa)

# %%
doc_folder_hotpotqa_q = '/Users/abhilashamangal/Documents/Semantic Search/data/hotpotqa/qrels'

# %%
len()

# %%
files_hotpotqa_q  = get_all_files(doc_folder_hotpotqa_q)

# %%
df_hotpotqa_q_t = pd.read_csv(files_hotpotqa_q[0],sep = '\t')

# %%
df_hotpotqa_q_t.head()

# %%
df_hotpotqa_q_t['score'].unique()

# %%



