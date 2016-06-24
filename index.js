var natural = require ('natural');
tokenizer = new natural.WordTokenizer();
classifier = new natural.BayesClassifier();

var reviews = require('./reviews.json');
var trainGroup = reviews.slice(0,400);
var testGroup = reviews.slice(401,600);

for(var i=0; i<trainGroup.length;i++)
{
	if(trainGroup[i].text){
		trainGroup[i].text = tokenizer.tokenize(trainGroup[i].text);
		trainGroup[i].text = trainGroup[i].text.join(" ");

		classifier.addDocument(trainGroup[i].text,trainGroup[i].class);
	}
}
classifier.train();

for(var i=0; i<testGroup.length; i++)
{
	if(testGroup[i].text){
		console.log(testGroup[i].class+ ' '+ classifier.classify(testGroup[i].text));
	}
}

