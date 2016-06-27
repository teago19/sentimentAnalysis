from imdbpie import Imdb
import json
imdb = Imdb(anonymize=True) # to proxy requests

reviews = imdb.get_title_reviews("tt0120338", max_results=2500)

classified_reviews = []

positive_reviews = [x for x in reviews if x.rating > 7]
negative_reviews = [x for x in reviews if x.rating < 5]

for i in range(0, 550):
  classified_reviews.append({
    'text': positive_reviews[i].text,
    'class': 'POSITIVE'
  })
  classified_reviews.append({
    'text': negative_reviews[i].text,
    'class': 'NEGATIVE'
  })

with open('result.json', 'w') as fp:
    json.dump(classified_reviews, fp)