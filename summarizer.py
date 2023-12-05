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


def sentenceSimilarity(sent1, sent2, stopwords=[]):
    sent1 = [w.lower() for w in sent1 if w.lower() not in stopwords]
    sent2 = [w.lower() for w in sent2 if w.lower() not in stopwords]

    all_words = list(set(sent1 + sent2))
    vector1 = [sent1.count(word) for word in all_words]
    vector2 = [sent2.count(word) for word in all_words]

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1, sentence1 in enumerate(sentences):
        for idx2, sentence2 in enumerate(sentences):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentenceSimilarity(sentence1, sentence2, stop_words)
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

summary = generate_summary("msft.txt", 2)
print("Summary:", summary)
