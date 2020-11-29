import os
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize

def get_target_document(directory_path):
    list_documents = {}
    list_query = []
    
    target_files = sorted(os.listdir(directory_path)) # get file list with sorted
    
    # iterate over target files
    for target_file in target_files:

        # check .txt only
        if target_file.lower().endswith('.txt'):

            # read text
            file_name = os.path.join(directory_path,target_file)
            file = open(file_name, "r")
            filedata = file.read()

            # get query from file name
            index_of_dot = target_file.index('.')
            query = target_file[:index_of_dot]
            list_query.append(query)
            # add dictionary with key is query and the value is the text
            list_documents[query] = filedata 
    
    return list_documents, list_query

def load_result_csv(filename):
    result_df = pd.read_csv(filename)
    result_df = result_df.sort_values(by = "Query", axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

    # get clean quiers
    queries = result_df['Query'].str.replace('\n','').str.lower().str.strip() # get clean queries

    return result_df, queries

def find_index_query(queries_df, query):
    """
    find index of queries data frame with query
    return: index
    """
    series = queries_df.str.find(query)
    for index, value in series.items():
        if value >= 0:
            return index

    return -1

def get_fscore(result_sentences, target_sentences):
    """
    Precision(P) is the number of sentences occurring in both system and ideal summaries divided by the number of sentences in the system summary
    Recall(R) is the numberof sentences occurring in both system and ideal summaries divided by the number ofsentences in the ideal summary
    """
    # get number of sentences
    num_res_sentences = len(result_sentences)
    num_target_sentences = len(target_sentences)

    # find same sentence 
    num_occur_sentences = 0
    for res_sentence in result_sentences:
        if res_sentence in target_sentences:
            num_occur_sentences += 1

    try:
        precision = num_occur_sentences / num_res_sentences
        recall = num_occur_sentences / num_target_sentences
        f_score = 2 * precision * recall / (precision + recall)
        
        return f_score
    
    except ZeroDivisionError:
        return 0

def calculate_performance(list_documents, list_query, result_df, queries):
    # iterate list query
    f_scores = []
    for query in list_query:
        
        # find index of query
        index_df = find_index_query(queries,query)
        
        # get result and target summarization
        result_summarization = result_df.loc[index_df]['Summarization']
        target_summarization = list_documents[query]

        # split it to token
        result_sentences = sent_tokenize(result_summarization)
        target_sentences = sent_tokenize(target_summarization)

        # get precision
        f_score = get_fscore(result_sentences, target_sentences)
        f_scores.append(f_score)
        # Debug
        print("\nQuery : {} \nTarget : \n{} \nResult : \n{} \nF-Score : {}".format(query,target_summarization,result_summarization,f_score))

    average_f_score = np.average(f_scores) # calculate average

    return average_f_score