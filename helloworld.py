# This Python file uses the following encoding: utf-8
from gensim import corpora, models, similarities
from gensim.models import ldamodel
from collections import defaultdict
from pprint import pprint  # pretty-printer
import string
import pyLDAvis
import pyLDAvis.gensim
from os import listdir
from os.path import isfile, join


stoplist = set(
    # 'dann will gut darf seinem dank nur allem uns ihren ihrem zwei drei seit seiner man mio euro neuen also war geht etwa gibt mehrere andere zf zwischen hat beim möglich ermöglicht richtige keine mehr alle hohen diesen kein noch neue nimmt diesem werden so misst usw zb um etc welches per eingehende diese zusätzlich aber als am an auch auf aus bei bin bis bist da dadurch daher darum das daß dass dein deine dem den der des dessen deshalb die dies dieser dieses doch dort du durch ein eine einem einen einer eines er es euer eure für hatte hatten hattest hattet hier	hinter ich ihr ihre im in ist ja jede jedem jeden jeder jedes jener jenes jetzt kann kannst können könnt machen mein meine mit muß mußt musst müssen müßt nach nachdem nein nicht nun oder seid sein seine sich sie sind soll sollen sollst sollt sonst soweit sowie und unser	unsere unter vom von vor wann warum was weiter weitere wenn wer werde werden werdet weshalb wie wieder wieso wir wird wirst wo woher wohin zu zum zur über'
        "iot p e f g fig data ← also used many see c – m s d one well - • can will use may a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our our ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves".split())
punctuations = set(string.punctuation)
digits = set(string.digits)


def getdocumentsfromfile():
    mypath = 'data/searched'
    documents = []
    for filename in listdir(mypath):
        if filename.endswith('.txt'):
            f = open(join(mypath, filename))
            fr = f.read()
            text = ''.join(ch for ch in fr if ch not in punctuations and ch not in digits)
            # print(text)
            documents.append(text)
    return documents

# def getdocumentsfromfile():
#     f = open('data/wearables-full.txt', 'r')
#     fr = f.read()
#     fr = ''.join(ch for ch in fr if ch not in punctuations and ch not in digits)
#     documents = fr.split('\n')
#     f.close()
#     f = open('data/rfid-full.txt', 'r')
#     fr = f.read()
#     fr = ''.join(ch for ch in fr if ch not in punctuations and ch not in digits)
#     documents += fr.split('\n')
#     f.close()
#     f = open('data/energyharvesting.txt', 'r')
#     fr = f.read()
#     fr = ''.join(ch for ch in fr if ch not in punctuations and ch not in digits)
#     documents += fr.split('\n')
#     f.close()
#     return documents


def preprocessdocuments():
    # remove common words and tokenize
    documents = getdocumentsfromfile()

    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
        ]
    #print(texts)
    # remove words that appear only once

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    #pprint(texts)

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
print(model.print_topics(3, 3))

vis_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
pyLDAvis.save_html(vis_data, 'e.html')

# print('Test: ')
# print(model[tfidf[dictionary.doc2bow(['smartes', 'armband', 'fitnessarmband', 'dienen', 'sms', 'emails', 'anzeigen'])]])
# print(model[tfidf[dictionary.doc2bow(['verfolgen', 'produktion', 'sicherstellen', 'richtigen', 'kunden', 'kunde', 'tag'])]])
# print(model[tfidf[dictionary.doc2bow(['sunpartner','transparente','solarfolie','entwickelt'])]])
