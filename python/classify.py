import itertools
import json
import math
import random

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.stem import PorterStemmer

observed_element = 0

def prepare_data(reviews):
    # run porter stemmer on every word
    stemmer = PorterStemmer()
    stem_text = lambda x: {'class': x['class'],
                           'text': stemmer.stem(x['text'])}

    # clean text and remove empty items
    reviews = filter(lambda x: x != {}, reviews)
    reviews = map(stem_text, reviews)

    print('classification: ' + reviews[observed_element]['class'] + '\n\n------------------------------------\n\n')

    print('stemming: ' + reviews[observed_element]['text'] + '\n\n------------------------------------\n\n')

    # remove stopwords
    reviews = map(remove_stop_words, reviews)

    print('stopwords: ' + reviews[observed_element]['text'] + '\n\n------------------------------------\n\n')

    # remove undesired patterns
    reviews = map(clean_text, reviews)

    print('elementos inuteis: ' + reviews[observed_element]['text'] + '\n\n------------------------------------\n\n')

    return reviews


def clean_text(item):
    patterns = [
        '\n', '.', '(', ')', ';', '-',
        '!', '?', '[', ']', '+', '|',
        ':', ',', '\"', '0', '1', '2',
        '3', '4', '5', '6', '7', '8', '9', '*'
    ]

    for pattern in patterns:
        item['text'] = item['text'].replace(pattern, ' ')

    return item


def remove_stop_words(item):
    stop_words = stopwords.words('english')
    item['text'] = " ".join(
        list(
            set(item['text'].split(' ')) - set(stop_words)
        )
    )
    return item


def word_feats(words):
    return dict([(word, True) for word in words])


def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def classify_and_evaluate(reviews, feature_extractor=word_feats):
    random.shuffle(reviews)

    pos_reviews = filter(lambda x: x['class'] == 'POSITIVE', reviews)
    neg_reviews = filter(lambda x: x['class'] == 'NEGATIVE', reviews)

    # get unique features
    pos_features = []
    neg_features = []
    for review in pos_reviews:
        split_reviews = review['text'].split(' ')
        split_reviews = [x for x in split_reviews if x]
        pos_features.append((feature_extractor(split_reviews), 'pos'))

    for review in neg_reviews:
        split_reviews = review['text'].split(' ')
        split_reviews = [x for x in split_reviews if x]
        neg_features.append((feature_extractor(split_reviews), 'neg'))

    # divide groups
    pos_offset = int(math.floor(len(pos_reviews) * 3 / 4))
    neg_offset = int(math.floor(len(neg_reviews) * 3 / 4))

    training = pos_features[:pos_offset] + neg_features[:neg_offset]
    testing = pos_features[pos_offset:] + neg_features[neg_offset:]

    # train classifier
    classifier = NaiveBayesClassifier.train(training)

    print 'treinada em %d reviews, testada em %d reviews' % (len(training), len(testing))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testing)
    classifier.show_most_informative_features()


f = open('../reviews.json', 'r')
reviews = json.loads(f.read())

reviews = prepare_data(reviews)

classify_and_evaluate(reviews)
