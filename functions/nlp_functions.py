import nltk
import re
import numpy as np
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk import SnowballStemmer
import pandas as pd

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

# removes a list of words (ie. stopwords) from a tokenized list.


def removeWords(listOfTokens, listOfWords):
    return [token for token in listOfTokens if token not in listOfWords]

# applies stemming to a list of tokenized words


def applyStemming(listOfTokens, stemmer):
    return [stemmer.stem(token) for token in listOfTokens]

# removes any words composed of less than 2 or more than 21 letters


def twoLetters(listOfTokens):
    twoLetterWord = []
    for token in listOfTokens:
        if len(token) <= 2 or len(token) >= 21:
            twoLetterWord.append(token)
    return twoLetterWord


def is_noun(pos): return pos[:2] == 'NN'


def keywords_inventory(dataframe, colonne='Description'):
    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys = []
    count_keywords = dict()
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s):
            continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(
            tokenized) if is_noun(pos)]

        for t in nouns:
            t = t.lower()
            racine = stemmer.stem(t)
            if racine in keywords_roots:
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1

    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k
                    min_length = len(k)
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]

    print("Nb of keywords in variable '{}': {}".format(
        colonne, len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords


def processCorpus(corpus, language):
    stopwords = nltk.corpus.stopwords.words(language)
    param_stemmer = SnowballStemmer(language)
    countries_list = [line.rstrip('\n') for line in open(
        '/home/rafael/Projetos/ds-projects/functions/lists/countries.txt')]  # Load .txt file line by line
    nationalities_list = [line.rstrip('\n') for line in open(
        '/home/rafael/Projetos/ds-projects/functions/lists/nationalities.txt')]  # Load .txt file line by line
    other_words = [line.rstrip('\n') for line in open(
        '/home/rafael/Projetos/ds-projects/functions/lists/stopwords_scrapmaker.txt')]  # Load .txt file line by line

    for document in corpus:
        index = corpus.index(document)
        # Replaces the ASCII '�' symbol with '8'
        corpus[index] = corpus[index].replace(u'\ufffd', '8')
        corpus[index] = corpus[index].replace(
            ',', '')          # Removes commas
        corpus[index] = corpus[index].rstrip(
            '\n')              # Removes line breaks
        # Makes all letters lowercase
        corpus[index] = corpus[index].casefold()

        # removes specials characters and leaves only words
        corpus[index] = re.sub('\W_', ' ', corpus[index])
        # removes numbers and words concatenated with numbers IE h4ck3r. Removes road names such as BR-381.
        corpus[index] = re.sub("\S*\d\S*", " ", corpus[index])
        # removes emails and mentions (words with @)
        corpus[index] = re.sub("\S*@\S*\s?", " ", corpus[index])
        # removes URLs with http
        corpus[index] = re.sub(r'http\S+', '', corpus[index])
        # removes URLs with www
        corpus[index] = re.sub(r'www\S+', '', corpus[index])

        listOfTokens = word_tokenize(corpus[index])
        twoLetterWord = twoLetters(listOfTokens)

        listOfTokens = removeWords(listOfTokens, stopwords)
        listOfTokens = removeWords(listOfTokens, twoLetterWord)
        listOfTokens = removeWords(listOfTokens, countries_list)
        listOfTokens = removeWords(listOfTokens, nationalities_list)
        listOfTokens = removeWords(listOfTokens, other_words)

        listOfTokens = applyStemming(listOfTokens, param_stemmer)
        listOfTokens = removeWords(listOfTokens, other_words)

        corpus[index] = " ".join(listOfTokens)
        corpus[index] = unidecode(corpus[index])

    return corpus
