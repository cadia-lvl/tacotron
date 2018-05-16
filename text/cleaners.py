import re
from unidecode import unidecode

# regular expressions

# matches >=1 spaces
_whitespace_re = re.compile(r'\s+')



# cleaners that can be applied in any order

def lowercase(text):
    '''
        Returns a lowercase version of the input string
        
        Input:
        text: a string
    '''
    return text.lower()

def transliterate(text):
    '''
        Returns a transliteration of the input string

        Input:
        text: a string

        Example:
        "Þjóðhöfðingi" -> "Thjodhofdingi"
    '''
    return unidecode(text)


def remove_whitespace(text):
    '''
        Returns a non-whitespace version of the input string

        Input:
        text: a string
    
        Example:
        "  this is  a string  " -> " this is a string "
    '''
    return re.sub(_whitespace_re, ' ', text)


# Cleaner collections : A list of multiple cleaners

def basic_cleaners(text):
    '''
        Takes in a string and applies
        * lowercasing every character
        * Removes whitespace
    '''
    text = lowercase(text)
    text = remove_whitespace(text)
    return text

def transliteration_cleaners(text):
    '''
        Takes in a string and applies
        * transliteration
        * lowercasing every character
        * Removes whitespace
    '''
    text = transliterate(text)
    text = lowercase(text)
    text = remove_whitespace(text)
    return text
