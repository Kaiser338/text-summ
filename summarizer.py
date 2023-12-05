import os
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    with open(file_name, "r") as file:
        filedata = file.read()

    sentences = filedata.replace("[^a-zA-Z]", " ").split(". ")
    if sentences[-1] == '':
        sentences.pop()

    processed_sentences = [sentence.split() for sentence in sentences]
    return processed_sentences

def sentence_similarity(sent1, sent2, stop_words=[]):
    sent1 = [w.lower() for w in sent1 if w.lower() not in stop_words]
    sent2 = [w.lower() for w in sent2 if w.lower() not in stop_words]

    all_words = list(set(sent1 + sent2))
    vector1 = [sent1.count(word) for word in all_words]
    vector2 = [sent2.count(word) for word in all_words]

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1, sentence1 in enumerate(sentences):
        for idx2, sentence2 in enumerate(sentences):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentence1, sentence2, stop_words)
    return similarity_matrix

def generate_summary(file_name, top_n=5):
    nltk.download("stopwords")
    stop_words = set(stopwords.words('english'))
    sentences = read_article(file_name)
    similarity_matrix = build_similarity_matrix(sentences, stop_words)
    similarity_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(similarity_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary = [' '.join(sent) for _, sent in ranked_sentences[:top_n]]
    return '. '.join(summary)

def list_txt_files_in_folder():
    txt_files = [f for f in os.listdir() if f.endswith(".txt")]
    if not txt_files:
        print("No TXT files found in the current folder.")
    else:
        print("Available TXT files:")
        for idx, file in enumerate(txt_files):
            print(f"{idx + 1}. {file}")
        return txt_files

def select_file_to_generate_summary(txt_files):
    while True:
        try:
            choice = int(input("Enter the number corresponding to the TXT file you want to summarize (0 to exit): "))
            if choice == 0:
                return None
            elif choice < 1 or choice > len(txt_files):
                print("Please enter a valid number.")
            else:
                return txt_files[choice - 1]
        except ValueError:
            print("Invalid input. Please enter a number.")

# List and select a TXT file in the current folder
while True:
    txt_files = list_txt_files_in_folder()
    if not txt_files:
        break

    selected_file = select_file_to_generate_summary(txt_files)
    if selected_file is None:
        break

    summary = generate_summary(selected_file, 2)
    print("\nSummary:", summary)
