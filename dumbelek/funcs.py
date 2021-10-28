import re
from TurkishStemmer import TurkishStemmer
from dumbelek.stopwordlist import StopWordList
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

class StopWordsTr:
    def get_stopwordList():
        '''
        Returns
        -------
        stopwordlist : list
            Returns all the stopwords as a list
        '''
        return StopWordList.get_stopwords()
    
    def remove_stopwords(text):
        '''
        Parameters
        ----------
        text : string

        Returns
        -------
        text : string
            Removes the Turkish stop-words from given text
        '''
        text = text.lower()
        to_return = ''
        word_list = text.split()
        for i in range(0,len(word_list),1):
            if word_list[i] in StopWordsTr.get_stopwordList():
                word_list[i] = None
            else:
                to_return += word_list[i] + " "
        return to_return
       
    def is_stopword(word):
        #Returns a boolean value whether the word is in the stop-word list
        '''
        Parameters
        ----------
        word : string

        Returns
        -------
        wordInList : bool
        Returns a boolean value whether the word is in the stop-word list
        '''
        return word in StopWordsTr.get_stopwordList()
    
class Cleaner:
    def remove_links(text):
        '''Takes a string and removes web links from it'''
        text = re.sub(r'http\S+', '', text)  # remove http links
        text = re.sub(r'bit.ly/\S+', '', text)  # remove bitly links
        # text = text.strip('[link]')  # remove [links]
        text = re.sub(r'#', '', text)
        return text
    
    def remove_users(text):
        '''Takes a string and removes retweet and @user information'''
        text = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)',
                       '', text)  # remove retweet
        text = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)',
                       '', text) # remove tweeted at
        text = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)',
                       '', text) # remove hashtags
        return text
    
    def clean_text(text):
        my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
        text = Cleaner.remove_users(text)
        text = Cleaner.remove_links(text)
        text = text.lower()  # lower case
        text = re.sub('[' + my_punctuation + ']+', ' ', text)  # strip punctuation
        text = re.sub('\s+', ' ', text)  # remove double spacing
        text = re.sub('([0-9]+)', '', text)  # remove numbers
        return text
    
    def clean_all(text): 
        '''
        Parameters
        ----------
        text : string

        Returns
        -------
        text : string
            Removes twitter punctuations such as: Retweets, Mentions, Hashtags
            Also removes short and long website links.
        '''
        
        text = Cleaner.clean_text(text)
        text = StopWordsTr.remove_stopwords(text)
        return text
    
class NgramCalc:
    def remove_duplicate_ngrams(df_ngram):
        for line, row in df_ngram.iterrows():
            threeGram = row['words']
            threeGramList = threeGram.split()
            if len(threeGramList) == 3:
                for l, r in df_ngram.iterrows():
                    twoGram = r['words']
                    if twoGram in threeGram and twoGram != threeGram:
                        df_ngram = df_ngram.drop(index=l)
    
        for line, row in df_ngram.iterrows():
            firstGram = row['words']
            firstGramFreq = row['frequency']
            firstGramList = firstGram.split()
            if len(firstGramList) > 2:
                for l, r in df_ngram.iterrows():
                    secondGram = r['words']
                    secondGramFreq = r['frequency']
                    secondGramList = secondGram.split()
                    if len(secondGramList) > 2 and firstGramFreq == secondGramFreq and firstGramList[0] == secondGramList[
                        1] and firstGramList[1] == secondGramList[2]:
                        df_ngram = df_ngram.drop(index=l)

        return df_ngram
    
    def get_ngrams_list(textList,topValues=None,ngram_range=(2,3),remove_duplicates=True,stem=False):
        '''
        Parameters
        ----------
        textList : list,
        topValues : int ##Amount of the most frequent ngrams you would like to get (if null, all ngrams are returned),
        ngram_range : tuple ##Size of the ngrams you would like to get (default is 2 and 3),
        remove_duplicates : bool ##Whether to override the bigger ngrams over the smaller chunks,
        stem : bool ##Whether to stem the words in the list of text
        
        Returns
        -------
        ngramDict : list 
        ## a list of dictionaries that contains the ngram text and frequencies
        '''
        c_vec = CountVectorizer(ngram_range=ngram_range,stop_words=StopWordsTr.get_stopwordList())
        textList=[Cleaner.clean_all(x) for x in textList]
        if stem:
            for text in textList:
                for word in text.split():
                    word = TurkishStemmer().stem(word)
        # matrix of ngrams
        try:
            ngrams = c_vec.fit_transform(textList)
        except ValueError:
            return []
        # count frequency of ngrams
        count_values = ngrams.toarray().sum(axis=0)
        
        # list of ngrams
        vocab = c_vec.vocabulary_
    
        df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
                                ).rename(columns={0: 'frequency', 1: 'words'})
    
    
        if remove_duplicates:
            df_ngram = NgramCalc.remove_duplicate_ngrams(df_ngram)
        if topValues:
            df_ngram = df_ngram.head(topValues)
        ngram_json = df_ngram.to_dict(orient='records')
    
        return ngram_json
    
    def get_ngrams_series(textSeries,topValues=None,ngram_range=(2,3),remove_duplicates=True,stem=False):
        '''
        Parameters
        ----------
        textList : pd.Series,
        topValues : int ##Amount of the most frequent ngrams you would like to get (if null, all ngrams are returned),
        ngram_range : tuple ##Size of the ngrams you would like to get (default is 2 and 3),
        remove_duplicates : bool ##Whether to override the bigger ngrams over the smaller chunks,
        stem : bool ##Whether to stem the words in the list of text
        
        Returns
        -------
        ngramDict : list ## a list of dictionaries that contains the ngram text and frequencies
        '''
        c_vec = CountVectorizer(ngram_range=ngram_range,stop_words=StopWordsTr.get_stopwordList())
        textList=[Cleaner.clean_all(x) for x in list(textSeries)]
        if stem:
            for text in textList:
                for word in text.split():
                    word = TurkishStemmer().stem(word)
        # matrix of ngrams
        try:
            ngrams = c_vec.fit_transform(textList)
        except ValueError:
            return []
        # count frequency of ngrams
        count_values = ngrams.toarray().sum(axis=0)
        
        # list of ngrams
        vocab = c_vec.vocabulary_
    
        df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)
                                ).rename(columns={0: 'frequency', 1: 'words'}).head(50)
    
        if remove_duplicates:
            df_ngram = NgramCalc.remove_duplicate_ngrams(df_ngram)
        if topValues:
            df_ngram = df_ngram.head(topValues)
        ngram_json = df_ngram.to_dict(orient='records')
    
        return ngram_json
    