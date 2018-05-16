pad        = '_'
eos        = '~'
# TODO: Questionable to include ';'
# TODO: If the basic cleaners is applied, the number of characters 
# in the alphabet dramatically decreases but this is not reflected 
# in generating the onehot representation.

_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
_isk_characters = 'ÁÐÉÍÓÚÝÞÆÖáðéíóúýþæö'
_numbers = '0123456789'

# Export all symbols:
chars = [pad, eos] + list(_characters) + list(_isk_characters) + list(_numbers)