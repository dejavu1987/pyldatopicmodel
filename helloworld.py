# This Python file uses the following encoding: utf-8
from gensim import corpora, models, similarities
from gensim.models import ldamodel
from collections import defaultdict
from pprint import pprint  # pretty-printer
import string
import pyLDAvis

stoplist = set(
    'dann will gut darf seinem dank nur allem uns ihren ihrem zwei drei seit seiner man mio euro neuen also war geht etwa gibt mehrere andere zf zwischen hat beim möglich ermöglicht richtige keine mehr alle hohen diesen kein noch neue nimmt diesem werden so misst usw zb um etc welches per eingehende diese zusätzlich aber als am an auch auf aus bei bin bis bist da dadurch daher darum das daß dass dein deine dem den der des dessen deshalb die dies dieser dieses doch dort du durch ein eine einem einen einer eines er es euer eure für hatte hatten hattest hattet hier	hinter ich ihr ihre im in ist ja jede jedem jeden jeder jedes jener jenes jetzt kann kannst können könnt machen mein meine mit muß mußt musst müssen müßt nach nachdem nein nicht nun oder seid sein seine sich sie sind soll sollen sollst sollt sonst soweit sowie und unser	unsere unter vom von vor wann warum was weiter weitere wenn wer werde werden werdet weshalb wie wieder wieso wir wird wirst wo woher wohin zu zum zur über'.split())
punctuations = set(string.punctuation)
digits = set(string.digits)


def getdocumentsfromfile():
    f = open('data/wearables.txt', 'r')
    fr = f.read()
    fr = ''.join(ch for ch in fr if ch not in punctuations and ch not in digits)
    documents = fr.split('\n')
    f.close()
    f = open('data/rfid.txt', 'r')
    fr = f.read()
    fr = ''.join(ch for ch in fr if ch not in punctuations and ch not in digits)
    documents += fr.split('\n')
    f.close()
    f = open('data/energyharvesting.txt', 'r')
    fr = f.read()
    fr = ''.join(ch for ch in fr if ch not in punctuations and ch not in digits)
    documents += fr.split('\n')
    f.close()
    return documents


def preprocessdocuments():
    # remove common words and tokenize
    documents = getdocumentsfromfile()

    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
        ]

    # remove words that appear only once

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    pprint(texts)

    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/trends.dict')  # store the dictionary, for future reference
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('/tmp/trends.mm', corpus)  # store to disk, for later use


preprocessdocuments()
dictionary = corpora.Dictionary.load('/tmp/trends.dict')
corpus = corpora.MmCorpus('/tmp/trends.mm')
tfidf = models.TfidfModel(corpus)
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=3)
print('Topics: ')
print(model.print_topics(3, 5))
print('Test: ')
print(model[tfidf[dictionary.doc2bow(['smartes', 'armband', 'fitnessarmband', 'dienen', 'sms', 'emails', 'anzeigen'])]])
print(model[tfidf[dictionary.doc2bow(['verfolgen', 'produktion', 'sicherstellen', 'richtigen', 'kunden', 'kunde', 'tag'])]])
print(model[tfidf[dictionary.doc2bow(['sunpartner','transparente','solarfolie','entwickelt'])]])
