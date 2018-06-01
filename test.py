import os

text = 'asdas dasd sa da dasda dad '
wav = 'asdad ads a daw da wad a'


root = '/home/atli/Desktop'
os.makedirs(os.path.join(root, 'wavs'), exist_ok=True)
os.makedirs(os.path.join(root, 'text'), exist_ok=True)
index_file = open('/home/atli/Desktop/myfile.test', 'a+')
index = sum(1 for line in open('/home/atli/Desktop/myfile.test')) + 1
wav_path = os.path.join(root, 'wavs', 'synth-%03d.wav' % index)
txt_path = os.path.join(root, 'text', 'text-%03d.txt' % index)
index_file.write(txt_path+'|'+wav_path+'|'+text+'\n')
txt_file = open(txt_path, 'w')
txt_file.write(text)
