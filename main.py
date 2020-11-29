import gensim
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from nltk.tokenize import word_tokenize,sent_tokenize,RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import networkx as nx
from collections import Counter
import math
import numpy as np
import operator
import os
import time
import csv
import pandas as pd
import datetime
from evaluation import *


nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('indonesian')

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# log filename
dt = datetime.datetime.now()
log_file_name = 'result/result_file_%s-%s-%s-%s-%s.csv' % (dt.day, dt.month, dt.year, dt.hour, dt.minute)

def read_article(file_name):
    """
    Generate clean sentences from the file
    Input : path file name
    Output : array of sentences
    """
    file = open(file_name, "r")
    filedata = file.readlines()
    sentences = []   
    for sentence in filedata:
        print("\n{} text: \n{}".format(file_name,sentence))
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" ")) # filter charachter only
    
    return sentences


def read_article_2(filename):
    """
    Generate clean sentences from the, split it to sentences
    Input : path file name
    Output : array of sentences
    """
    file = open(filename, "r")
    filedata = file.readlines()
    sentences = sent_tokenize(filedata[0])
    return sentences


def merge_sentences(sentences):
    """
    Merge sentences to one array
    """
    full_sentences = []
    for sentence in sentences:
        for arr_word in sentence:
            full_sentences.append(arr_word)
    return full_sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    
    return similarity_matrix


def preprocessing(data):
    """
    Do pre processing data, tokenization, remove stop words, and stemming
    Input : data (string)
    Output : array of words
    """
    #tokenizer = RegexpTokenizer(r'\w+') # allow charachter only
    #words = tokenizer.tokenize(data) # tokenize : convert to words
    words = word_tokenize(data)
    # remove stop words & stemming
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(stemmer.stem(word)) # append to new words with stemming
    
    if '' in new_words: new_words.remove('') # remove space from list
    #print("Preprocessing : {}".format(new_words))
    return new_words


def get_cosine(vec1, vec2):
    intersection = set(vec1) & set(vec2.keys()) # find the same word
    numerator = sum([vec1[x] * vec2[x] for x in intersection]) # iterate intersection, and total the times frequency

    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) # iterate the dictionary, and square the elements then sum it
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) # iterate the vector2 and square the elements then sum it
    denominator = math.sqrt(sum1) * math.sqrt(sum2) # square root the sum1 times square root the sum2

    if not denominator:
        # the denominator is zero
        return 0.0
    else:
        return numerator / denominator


def text_to_vector(text):
    words = preprocessing(text)
    return Counter(words)


def get_similarity(sentences, query):
    cosine_mat=np.zeros(len(sentences)+1)
   
    vector_query = text_to_vector(query) # calculate query vector
    row=0
    # calculate cosine value for every sentence 
    for sentence in sentences:
        maxi=0
        vector_sentence = text_to_vector(sentence) # get vector for the sentence
       
        cosine = get_cosine(vector_sentence, vector_query) # calculate cosine similarity
        
        # prevent from negative value
        if(maxi<cosine):
            maxi=cosine
               
        cosine_mat[row]=maxi # set to sentence index
             
        row+=1
        
    similarity = sum(cosine_mat) # sum all cosine array
        
    return similarity

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# csvwrite
def write_tocsv(file_name, dataframe) :
    """
    Write result data to csv file
    :param data:
    :return:
    """
    print("\nSaved result to {}\n".format(file_name))
    dataframe.to_csv(file_name, mode='a', header=False,index=False)

def main():
    '''
    Main code here
    '''
    # Step 1 : Define directories
    folder_root = 'data set'
    sub_folder = 'Tafsir al-Misbah'
    directory_path = os.path.join(folder_root,sub_folder) # join folder root and label
    files = sorted(os.listdir(directory_path)) # get file list with sorted

    # Step 2 : Load Query File
    queries = open('data set/Query.txt', "r").readlines()

    # Step 3 : Iterate over queries
    document_limit = 15
    
    for query in queries:
        print('\n ============================= \n')
        print("Query : {}".format(query))
        # iterate for all files
        list_similarity = {}
        inc = 0
        log_data = []
        log_data.append(query) # save query

        for file in files:
            # check for .txt only
            if file.lower().endswith('.txt'):
                # read article
                file_name = os.path.join(directory_path,file) # get full path
                sentences = read_article_2(file_name) # read file
                similarity = get_similarity(sentences,query) # get similarity
                
                # check if any similarity
                if (similarity > 0.0):
                    list_similarity[file] = similarity # set similarity

                # check if length list_similarity more than 15, then break instead check all file 
                if len(list_similarity) >= document_limit:
                    break # break

            printProgressBar(inc + 1, len(files), prefix = 'Progress:', suffix = 'Complete', length = 50)
            
            inc += 1
        
        #print("Similarity : \n {}".format(list_similarity))
        sorted_similarity = dict(sorted(list_similarity.items(), key=operator.itemgetter(1),reverse=True)) # sorted
        print("\nSorted Cosine Similarity : \n {}".format(sorted_similarity))
        
        # Step 4 : After get list of file, then do text summarization
        # check for any similarity
        if(len(sorted_similarity) >0):
            log_data.append(len(sorted_similarity)) # number of document
            log_data.append(sorted_similarity) # save sorted similarity
            # iterate over sorted_similarity, get document text
            documents = []
            list_file = []
            for key in sorted_similarity:
                file_name = os.path.join(directory_path,key) # get full path
                
                # read file
                document = read_article(file_name)
                documents.append(document)

                # append to file list for log
                list_file.append(key)

            # merge document to one documents
            full_sentences = merge_sentences(documents)

            # similarity matrix
            sentence_similarity_matrix = build_similarity_matrix(full_sentences,stop_words)
            
            # page rank sentences
            sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
            scores = nx.pagerank(sentence_similarity_graph)
        
            # sort the rank
            ranked_sentence = sorted(((scores[i],list_file[i],sentence) for i, sentence in enumerate(full_sentences)), reverse=True)
            log_data.append(ranked_sentence) # save score to csv

            # summarize the text
            top_n = 7 # rank 7
            summarize_text = []
            for i in range(top_n):
                summarize_text.append(" ".join(ranked_sentence[i][2]))

            # convert to string
            str_summarize = ' '.join(summarize_text)
            print("\nHasil Rangkuman {} rank teratas : \n{}".format(top_n, str_summarize))
            log_data.append(str_summarize) # save summarize to csv

        else:
            log_data.append("Not found") # save sorted similarity
            # no similarity found
            print("\nNo result in query...")

        # log data write to csv
        log_datas = pd.DataFrame([log_data])
        write_tocsv(log_file_name,log_datas)

def evaluation(log_file):
    # file list in folder
    folder_root = "data set"
    sub_folder = "Target Summarization"
    directory_path = os.path.join(folder_root,sub_folder) # join folder root and label
    list_document, list_query = get_target_document(directory_path)
    result_df, queries = load_result_csv(log_file)

    average_f_score = calculate_performance(list_document, list_query, result_df, queries)

    print("\nAverage F-Score: {} ".format(average_f_score))


if __name__ == "__main__":

    # log header
    print("Log file : {}".format(log_file_name))
    header_col = ['Query','Total','Similarity Queries','Rank Documents','Summarization']
    write_tocsv(log_file_name,pd.DataFrame([header_col]))
    
    main()

    print("\n-- Evaluation --")
    evaluation(log_file_name)