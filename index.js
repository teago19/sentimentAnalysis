var natural = require ('natural');
tokenizer = new natural.WordTokenizer();
classifier = new natural.BayesClassifier();

var reviews = require('./reviews.json');

//Run the PorterStemmer trought the texts
for(var i=0; i<reviews.length;i++)
{
	if(reviews[i].text){
		reviews[i].text = natural.PorterStemmer.stem(reviews[i].text);
	}
}

//Divide the reviews in two groups
var trainingGroup = reviews.slice(0,400);
var testGroup = reviews.slice(401,600);

//classifying the training group
for(var i=0; i<trainingGroup.length;i++)
{
	if(trainingGroup[i].text){
		trainingGroup[i].text = tokenizer.tokenize(trainingGroup[i].text);

		classifier.addDocument(trainingGroup[i].text,trainingGroup[i].class);
	}
}
classifier.train();

for(var i=0; i<testGroup.length; i++)
{
	if(testGroup[i].text){
		testGroup[i].text = tokenizer.tokenize(testGroup[i].text);

		console.log(classifier.getClassifications(testGroup[i].text))
		console.log(testGroup[i].class+ ' '+ classifier.classify(testGroup[i].text)+'\n');
	}
}

