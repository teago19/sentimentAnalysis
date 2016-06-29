import json
import math
import random

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def word_feats(words):
    return dict([(word, True) for word in words])


def clean_text(item):
    patterns = [
        '\n', '.', '(', ')', ';', '-',
        '!', '?', '[', ']', '+', '|',
        ':', ',', '\"'
    ]

    for pattern in patterns:
        item['text'] = item['text'].replace(pattern, ' ')

    return item


stop_words = stopwords.words('english')


def remove_stop_words(item):
    item['text'] = " ".join(
        list(
            set(item['text'].split(' ')) - set(stop_words)
        )
    )
    return item


f = open('../reviews.json', 'r')
reviews = json.loads(f.read())

# run porter stemmer through all texts
stemmer = PorterStemmer()
stem_text = lambda x: {'class': x['class'],
                       'text': stemmer.stem(x['text'])}

# clean text and remove empty items
reviews = filter(lambda x: x != {}, reviews)
reviews = map(stem_text, reviews)

# remove stopwords
reviews = map(remove_stop_words, reviews)

reviews = map(clean_text, reviews)
# reviews = map(remove_stop_words, reviews)

random.shuffle(reviews)

pos_reviews = filter(lambda x: x['class'] == 'POSITIVE', reviews)
neg_reviews = filter(lambda x: x['class'] == 'NEGATIVE', reviews)

# get unique features
pos_features = []
neg_features = []
for review in pos_reviews:
    pos_features.append((word_feats(review['text'].split(' ')), 'pos'))

for review in neg_reviews:
    neg_features.append((word_feats(review['text'].split(' ')), 'neg'))

# divide groups
#   training: 0~400
#   testing: 401~600

neg_features = list(neg_features)
pos_features = list(pos_features)

pos_offset = int(math.floor(len(pos_reviews) * 3 / 4))
neg_offset = int(math.floor(len(neg_reviews) * 3 / 4))

training = pos_features[:pos_offset] + neg_features[:neg_offset]
testing = pos_features[pos_offset:] + neg_features[neg_offset:]

# train classifier

classifier = NaiveBayesClassifier.train(training)

print 'accuracy:', nltk.classify.util.accuracy(classifier, testing)
classifier.show_most_informative_features()
