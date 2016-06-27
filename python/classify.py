from nltk.classify import NaiveBayesClassifier
from nltk.stem import PorterStemmer
import json, re, math
import random
from nltk.corpus import stopwords

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
stem_text = lambda x: { 'class': x['class'], 
                        'text': stemmer.stem(x['text']) }

# clean text and remove empty items
reviews = filter(lambda x: x != {}, reviews)
reviews = map(stem_text, reviews)

#remove stopwords
reviews = map(remove_stop_words, reviews)

reviews = map(clean_text, reviews)
#reviews = map(remove_stop_words, reviews)

pos_reviews = filter(lambda x: x['class'] == 'POSITIVE', reviews)
neg_reviews = filter(lambda x: x['class'] == 'NEGATIVE', reviews)

# get unique features
pos_feat_set = set()
neg_feat_set = set()
#pos_features = []
#neg_features = []
for review in pos_reviews:
    #pos_features.append(({'text':review['text']}, 'pos'))
    for feat in review['text'].split(' '):
        pos_feat_set.add(feat)

for review in neg_reviews:
    #neg_features.append(({'text':review['text']}, 'neg'))
    for feat in review['text'].split(' '):
        neg_feat_set.add(feat)

# label features for tests
pos_features = []
neg_features = []
for review in pos_reviews:
    for feat in pos_feat_set:
        pos_features.append(({'text':feat},'pos'))

for review in neg_reviews:
    for feat in neg_feat_set:
        neg_features.append(({'text':feat},'neg'))

# divide groups
#   training: 0~400
#   testing: 401~600

neg_features = list(neg_features)
pos_features = list(pos_features)

pos_offset = int(math.floor(len(pos_reviews)*3/4))
neg_offset = int(math.floor(len(neg_reviews)*3/4))

training = pos_features[:pos_offset] + neg_features[:neg_offset]
testing = pos_features[pos_offset:] + neg_features[neg_offset:]

random.shuffle(training)
random.shuffle(testing)

# train classifier
training_words = []
for item in training:
    for word in item[0]['text'].split(' '):
        training_words.append({'class':item[1], 'text':word})

classifier = NaiveBayesClassifier.train(training)

right, wrong = 0, 0
for item in testing:
    classification = classifier.classify(item[0])
    print classification
    if classification == item[1]:
        right += 1
    else:
        wrong += 1
