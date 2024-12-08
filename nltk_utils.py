import nltk  # type: ignore
nltk.download('punkt_tab')

from nltk.stem.porter import PorterStemmer  # type: ignore

stemmer = PorterStemmer()

# Tokenize a sentence into words
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Stem a word (reduce it to its root form)
def stem(word):
    return stemmer.stem(word.lower())

# Create a bag of words representation for a tokenized sentence
def bag_of_words(tokenized_sentence, all_words):
    # Stem each word in the sentence
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    # Initialize a bag with zeros for each word in all_words
    bag = [1 if word in tokenized_sentence else 0 for word in all_words]
    return bag

# Example usage
# a = "tôi tên là Hiền"
# print("Original sentence:", a)

# a = tokenize(a)
# print("Tokenized:", a)

# Define a vocabulary of all words and create the bag of words
# all_words = ["tôi", "tên", "là", "Hiền"]
# bag = bag_of_words(a, all_words)
# print("Bag of words:", bag)
