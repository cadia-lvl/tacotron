from text.text_tools import text_to_onehot


def synthesize(self, text):
    onehot = text_to_onehot(text)
    