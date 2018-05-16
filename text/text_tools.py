from text.characters import chars, pad, eos
from text import cleaners

import numpy as np

_char_to_ind = {c: i for i, c in enumerate(chars)}
_ind_to_char = {i: c for i, c in enumerate(chars)}

def text_to_onehot(text, cleaner_name='basic_cleaners'):
    '''
        Converts a string to it's onehot representation 
        according to the given character sequence.
        
        Input:
        text: A string

        Output:
        list of indexes into the given character sequence
        for each character in the string
    '''
    text = _clean_text(text, cleaner_name)
    onehot = [_char_to_onehot(c) for c in text if _should_keep(c)]
    onehot.append(_char_to_onehot(eos))
    return np.asarray(onehot)

def onehot_to_text(onehot):
    '''
        Converts a onehot list to it's string representation 
        according to the given character sequence.
        
        Input:
        onehot: a list of ints

        Output:
        The represented strings
    '''
    return ''.join([_onehot_to_char(o) for o in onehot])

def _char_to_onehot(c):
    '''
        Converts a single character to it's onehot representation

        Input:
        c: A character
    '''
    return _char_to_ind[c]

def _onehot_to_char(o):
    '''
        Converts a single integer to it's character representation

        Input:
        o: An integer
    '''
    return _ind_to_char[o]


def _clean_text(text, cleaner_name):
    '''
        Perform the text cleaning with the given cleaner. 
        The cleaner_name can reference either a single cleaner 
        or a collection of cleaners
    '''
    cleaner = getattr(cleaners, cleaner_name)
    return cleaner(text)

def _should_keep(c):
    '''
        Returns True if the character is in the alphabet
        and is not the special characters to indicate padding
        or end-of-string, otherwise False
    '''
    return c in _char_to_ind and c is not pad and c is not eos
