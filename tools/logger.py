from urllib.request import Request, urlopen
import json
import atexit
from datetime import datetime

class TrainingLogger:
    '''
        Keeps a log for training runs
        * output_file: The path to the file to which 
        the log will be written
        * slack_url: An optional webhook string for sending
        logs to slack in real time
     '''
    def __init__(self, output_file, slack_url=None):
        self._file = open(output_file, 'a')
        self._slack_url = slack_url
        self._date_format = '%d-%-m-%Y %H:%M:%S'
        atexit.register(self._close)
        self.line()
        self.log('Starting a training run')
        self.line()

    def log(self, msg, title=True, slack=False):
        '''
            Prints a string to standard out, saves it to the
            given output file and reports it optionally to a 
            slack channel.

            Input:
            msg: A string
            slack: A boolean
        '''
        date = datetime.now().strftime(self._date_format)
        if title:
            logged_msg = '[%s]: %s \n' % (date, msg) 
        else:
            logged_msg = '%s \n' % msg             
        print(logged_msg)
        self._file.write(logged_msg)
        if slack:
            self._slack_msg(date, msg)

    def _slack_msg(self, title, body):
        '''
            Sends a message with a title and and a body
            to the set slack webhook
        '''
        req = Request(self._slack_url)
        req.add_header('Content-Type', 'application/json')
        urlopen(req, json.dumps({
            'username': 'Tacotron',
            'icon_emoji': ':taco:',
            'text': '*%s*: %s' % (title, body)
        }).encode())
    
    def line(self):
        '''
            Log a line for readability
        '''
        self.log('------------------------------------', 
            title=False, slack=False)

    def _close(self):
        '''
            Indicates an end of logging and closes
            the opened log file
        '''
        self.line()
        self.log("Training is over, goodbye.")
        self.line()
        self._file.close()