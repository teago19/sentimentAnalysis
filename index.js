var natural = require ('natural');

var trainGroup = [
	{
		sentence: "i didn't like it",
		classification: "negative"
	},
	{
		sentence: "i expected more",
		classification: "negative"
	},
	{
		sentence: "i liked it",
		classification: "positive"
	},
	{
		sentence: "i liked it",
		classification: "positive"
	},
	{
		sentence: "i enjoyed it",
		classification: "positive"
	}
]

var testGroup =[
	{
		sentence: "didn't like it",
		classification: "negative"
	},
	{
		sentence: "it was boring",
		classification: "negative"
	},
	{
		sentence: "liked it",
		classification: "positive"
	},
	{
		sentence: "awesome!",
		classification: "positive"
	}
]


tokenizer = new natural.WordTokenizer();
console.log(tokenizer.tokenize(""));

classifier = new natural.BayesClassifier();
for(var i=0; i<trainGroup.length; i++){
	classifier.addDocument(trainGroup[i].sentence,trainGroup[i].classification);
}

classifier.train();

