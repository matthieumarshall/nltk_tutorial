##########################
# Tokenising words
##########################

from nltk.tokenize import sent_tokenize, word_tokenize

# we create a variable with some text to test our nltk on
example_text = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is " \
               "pinkish-blue. You shouldn't eat cardboard."

# we use the sent_tokenize function to output the sentences in our text
print(sent_tokenize(example_text))

# we use the word_tokenize function to output the words in our text
print(word_tokenize(example_text))

##########################
# Stop words
##########################

# from nltk.corpus we import stopwords
# stopwords are the words which have little meaning in text such as "this", "a", "had"
from nltk.corpus import stopwords

# we gather the english stop_words for use
stop_words = set(stopwords.words("english"))

# we tokenise our example_text
words = word_tokenize(example_text)

# we initialise a list in which we will assemble our words excluding the stop words
filtered_text = []

# we assemble our list of words from the example_text that are not stop words
for w in words:
    if w not in stop_words:
        filtered_text.append(w)

# we print these out for looking at, notice that you can still understand the general meaning of the text without
# the stop words
print(filtered_text)

# a one line version of the above filtering is
filtered_text = [w for w in words if not w in stop_words]

##########################
# Stemming words
##########################

# stemming removes afixes from words which still provide them with the same meaning
# for example taken, took, taking all more or less have the same meaning
# stemming helps us to consolidate words with the same meaning together

# PorterStemmer is a stemming algorithm that can be used for this
from nltk.stem import PorterStemmer

# we initialise our algorithm
ps = PorterStemmer()

# we create a list of example words that more or less have the same meaning
example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

# for each word in our list, we print the stem of the words
for w in example_words:
    print(ps.stem(w))

# we create another example text to analyse
new_text = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythoned " \
           "at least once"

# we tokenise the above
words = word_tokenize(new_text)

# again, we print out the stemmed words in our sentence to see the effect
for w in words:
    print(ps.stem(w))

# in reality, when using something like wordnet in nltk you won't have to do or think about the stemming
# but it is an important thing to think about

##########################
# Part of speech tagging
##########################

import nltk
# we import a text of state of the union speeches in the past
from nltk.corpus import state_union
# we import the PunkSentenceTokenizer, which is an unsupervised sentence tokenizer
# it comes pre-trained, although you can re-train it
from nltk.tokenize import PunktSentenceTokenizer

# we extract the text from Bush's speech in 2006
sample_text = state_union.raw("2006-GWBush.txt")

# we extract the text from the previous year to train on
train_text = state_union.raw("2005-GWBush.txt")

# we train our tokenizer on this training dataset
custom_sentence_tokenizer = PunktSentenceTokenizer(train_text)

# we sentence tokenize our text
tokenized = custom_sentence_tokenizer.tokenize(sample_text)

def process_content():
    try:
        # for each sentence
        for i in tokenized:
            # we extract the list of words from our sentence
            words = word_tokenize(i)
            # we tag the words in our sentence with the part of speech that they are
            tagged = nltk.pos_tag(words)
            # we then print out the words and what they are tagged with
            print(tagged)

    except Exception as e:
        print(str(e))

# we call the above function which does the part of speech tagging
process_content()

# Looking at the output and comparing with the definitions of the tags below,
# it all appears to make sense with nouns being tagged as nouns, verbs as verbs, etc
# note that it does get confused when pronouns are not capitalised

# the meaning of the part of speech tags is:
# POS tag list:
#
# CC	coordinating conjunction
# CD	cardinal digit
# DT	determiner
# EX	existential there (like: "there is" ... think of it like "there exists")
# FW	foreign word
# IN	preposition/subordinating conjunction
# JJ	adjective	'big'
# JJR	adjective, comparative	'bigger'
# JJS	adjective, superlative	'biggest'
# LS	list marker	1)
# MD	modal	could, will
# NN	noun, singular 'desk'
# NNS	noun plural	'desks'
# NNP	proper noun, singular	'Harrison'
# NNPS	proper noun, plural	'Americans'
# PDT	predeterminer	'all the kids'
# POS	possessive ending	parent\'s
# PRP	personal pronoun	I, he, she
# PRP$	possessive pronoun	my, his, hers
# RB	adverb	very, silently,
# RBR	adverb, comparative	better
# RBS	adverb, superlative	best
# RP	particle	give up
# TO	to	go 'to' the store.
# UH	interjection	errrrrrrrm
# VB	verb, base form	take
# VBD	verb, past tense	took
# VBG	verb, gerund/present participle	taking
# VBN	verb, past participle	taken
# VBP	verb, sing. present, non-3d	take
# VBZ	verb, 3rd person sing. present	takes
# WDT	wh-determiner	which
# WP	wh-pronoun	who, what
# WP$	possessive wh-pronoun	whose
# WRB	wh-abverb	where, when

##########################
# Chunking with nltk
##########################

# Chunking is a way of correctly linking adjectives with nouns within a sentence
# the downside of this is that you can only chunk words that are near each other
# Chunking uses a mix of regular expressions and Parts of Speech tags.

# we now partially re-use some code from earlier

def process_content_chunking():
    try:
        # for each sentence
        for i in tokenized:
            # we extract the list of words from our sentence
            words = word_tokenize(i)
            # we tag the words in our sentence with the part of speech that they are
            tagged = nltk.pos_tag(words)

            # we create a chunkGram looking for phrases that might contain an adverb followed by a verb, followed by
            # a pro-noun an a noun
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>?}"""

            # we create our parser which will use the regex chunk gram above
            chunkParser = nltk.RegexpParser(chunkGram)

            # we chunk our tagged text
            chunked = chunkParser.parse(tagged)

            # in this output we will see a bunch of nouns have been found by chunk
            print(chunked)

            # the above won't make much sense, but using chunked.draw will look a bit more human understandable
            # as it graphically shows you what noun phrases have been extracted from sentences
            chunked.draw()

    except Exception as e:
        print(str(e))

# we call the above chunking function
process_content_chunking()