import re

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
nltk.download("wordnet")
nltk.download("stopwords")
train_disaster = pd.read_csv(
    "train.csv", sep=";", encoding="utf-8", on_bad_lines="skip"
)

print(train_disaster["target"].value_counts())

plt.hist(train_disaster["target"])
plt.show()

# We can see that  class 1 is almost 50 percent of
# class 0 hence we shall not proceed with class balancing

# # 1. Data cleaning on text data

print("Check for null values")
print(train_disaster.isna().sum())


# define a function to clean the text data
# use re. sub() function which is used to replace occurrences
# of a particular sub-string with another sub-string.


def text_cleaning(text):
    text = text.lower()  # make in lower case
    text = re.sub(r"\[.*?@\]", "", text)  # remove text
    text = re.sub(r"\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)  # remove words containing numbers
    text.lstrip("$")  # removes $ sign from start of string
    text.strip()
    text = re.sub(r"[!@#$]", "", text)  # replace given characters from string
    return text


print(train_disaster.head(10))
train_disaster["text"] = \
    train_disaster["text"].apply(lambda x: text_cleaning(x))
print(train_disaster.head(10))


# """Tokenization"""


def tokenization(text):
    tokens = re.split("W+", text)
    return tokens


train_disaster["tokenized_text"] = train_disaster["text"].apply(
    lambda x: tokenization(x)
)
print(train_disaster["tokenized_text"])

# """Stemming"""

porter = PorterStemmer()


def stemming(text):
    stemtext = [porter.stem(i) for i in text]
    return stemtext


train_disaster["stemmed_text"] = train_disaster["tokenized_text"].apply(
    lambda x: stemming(x)
)
print(train_disaster["stemmed_text"])

# """Lemmatization"""

lemma = WordNetLemmatizer()


def lemmatization(text):
    lem_text = [lemma.lemmatize(i) for i in text]
    return lem_text


train_disaster["lemmatized_text"] = train_disaster["tokenized_text"].apply(
    lambda x: lemmatization(x)
)
print(train_disaster["lemmatized_text"])

# Stop words removal

stopwords = nlp.Defaults.stop_words
stopwords = nltk.corpus.stopwords.words("english")

stopwords[0:300] = [
    "whereupon",
    "n‘t",
    "whoever",
    "ca",
    "serious",
    "seemed",
    "been",
    "few",
    "which",
    "there",
    "myself",
    "part",
    "seeming",
    "indeed",
    "call",
    "another",
    "namely",
    "show",
    "used",
    "for",
    "sometime",
    "wherever",
    "bottom",
    "ever",
    "fifteen",
    "ten",
    "top",
    "done",
    "noone",
    "not",
    "yourself",
    "beyond",
    "afterwards",
    "move",
    "more",
    "most",
    "therein",
    "back",
    "'ve",
    "my",
    "himself",
    "‘ll",
    "any",
    "perhaps",
    "something",
    "last",
    "until",
    "anyhow",
    "nobody",
    "our",
    "hereby",
    "re",
    "hers",
    "does",
    "put",
    "every",
    "into",
    "such",
    "they",
    "everywhere",
    "one",
    "always",
    "has",
    "full",
    "anyway",
    "third",
    "us",
    "it",
    "towards",
    "almost",
    "on",
    "out",
    "her",
    "as",
    "might",
    "same",
    "your",
    "me",
    "hundred",
    "together",
    "the",
    "already",
    "an",
    "eight",
    "mostly",
    "have",
    "further",
    "only",
    "using",
    "what",
    "whereas",
    "though",
    "name",
    "being",
    "became",
    "regarding",
    "side",
    "moreover",
    "under",
    "did",
    "whether",
    "amongst",
    "that",
    "whence",
    "when",
    "we",
    "empty",
    "well",
    "herself",
    "eleven",
    "whither",
    "say",
    "him",
    "even",
    "off",
    "against",
    "give",
    "below",
    "beforehand",
    "really",
    "'ll",
    "itself",
    "made",
    "thus",
    "toward",
    "his",
    "‘d",
    "you",
    "get",
    "whole",
    "a",
    "would",
    "ours",
    "becomes",
    "nevertheless",
    "many",
    "unless",
    "throughout",
    "either",
    "over",
    "these",
    "and",
    "so",
    "them",
    "’ll",
    "those",
    "since",
    "somehow",
    "’re",
    "alone",
    "neither",
    "without",
    "forty",
    "cannot",
    "make",
    "he",
    "twelve",
    "front",
    "in",
    "none",
    "down",
    "after",
    "was",
    "thereupon",
    "keep",
    "around",
    "go",
    "however",
    "no",
    "becoming",
    "yourselves",
    "else",
    "just",
    "between",
    "yet",
    "whereby",
    "’m",
    "others",
    "who",
    "former",
    "had",
    "amount",
    "among",
    "everyone",
    "herein",
    "two",
    "nor",
    "other",
    "could",
    "thereafter",
    "still",
    "thereby",
    "anyone",
    "because",
    "before",
    "rather",
    "will",
    "hereafter",
    "latterly",
    "‘m",
    "how",
    "may",
    "three",
    "across",
    "do",
    "'m",
    "become",
    "whom",
    "up",
    "along",
    "each",
    "due",
    "sometimes",
    "anything",
    "within",
    "is",
    "several",
    "should",
    "latter",
    "themselves",
    "are",
    "by",
    "whereafter",
    "she",
    "someone",
    "nothing",
    "nowhere",
    "behind",
    "or",
    "too",
    "twenty",
    "wherein",
    "be",
    "except",
    "once",
    "enough",
    "besides",
    "first",
    "am",
    "'s",
    "quite",
    "anywhere",
    "from",
    "can",
    "about",
    "onto",
    "’s",
    "this",
    "then",
    "than",
    "all",
    "ourselves",
    "at",
    "while",
    "also",
    "‘re",
    "if",
    "five",
    "upon",
    "yours",
    "least",
    "very",
    "although",
    "where",
    "less",
    "above",
    "nine",
    "much",
    "’d",
    "hence",
    "of",
    "‘ve",
    "whose",
    "’ve",
    "meanwhile",
    "see",
    "doing",
    "per",
    "elsewhere",
    "their",
    "mine",
    "whatever",
    "via",
    "to",
    "were",
    "some",
    "thence",
    "various",
    "‘s",
    "here",
    "why",
    "please",
    "thru",
    "through",
    "seems",
    "take",
    "again",
    "during",
    "seem",
    "six",
    "n't",
    "formerly",
    "sixty",
    "'re",
    "four",
    "n’t",
    "but",
    "everything",
    "whenever",
    "'d",
    "often",
    "never",
    "with",
    "next",
    "hereupon",
    "otherwise",
    "i",
    "somewhere",
    "both",
    "beside",
    "fifty",
    "therefore",
    "its",
    "now",
    "own",
    "must",
]


def stopword_removal(text):
    removed_stopwords = [i for i in text if i not in stopwords]
    return removed_stopwords


train_disaster["final_cleaned_text"] = train_disaster["lemmatized_text"].apply(
    lambda x: stopword_removal(x)
)
print(train_disaster["final_cleaned_text"])


y = train_disaster["target"]
x = train_disaster["final_cleaned_text"].astype(str)

train_disaster["final_cleaned_text"] = [
    " ".join(i) for i in train_disaster["final_cleaned_text"].values
]

x_train, x_val, y_train, y_val = \
    train_test_split(x, y, test_size=0.25, random_state=40)

# Using Naive Bayes and vectorization

pipeline = Pipeline(
    [
        ("Tfidf", TfidfVectorizer()),
        ("naivebayes", MultinomialNB(alpha=1.0,
                                     fit_prior=True,
                                     class_prior=None)),
    ]
)

print(type(y_train))
pipeline.fit(x_train, y_train)

predict = pipeline.predict(x_val)

print(accuracy_score(y_val, predict))

accuracy = accuracy_score(y_val, predict)

print(classification_report(y_val, predict))

test_disaster = pd.read_csv(
    "test.csv",
    sep=";",
    encoding="utf-8",
    names=["id", "keyword", "location", "text", "target"],
    on_bad_lines="skip",
)

test_disaster = test_disaster.fillna(" ")
xtest = test_disaster["text"]

predict_test = pipeline.predict(xtest)

test_disaster["target"] = predict_test

final_test_disaster = test_disaster[["id", "target"]]

print(final_test_disaster)

final_test_disaster.to_csv("sample_submission.csv", index=False)
