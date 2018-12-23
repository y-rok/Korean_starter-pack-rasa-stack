import re

from konlpy.tag import Mecab

from rasa_nlu.components import Component
from rasa_nlu.tokenizers import Tokenizer, Token

class KoreanTokenizer(Tokenizer, Component):

    name = "tokenizer_whitespace"
    provides = ["tokens"]

    def __init__(self, component_config=None):
        self.mecab=Mecab()
        super(KoreanTokenizer, self).__init__(component_config)


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]

        # there is space or end of string after punctuation
        # because we do not want to replace 10.000 with 10 000
        words = re.sub(r'[.,!?]+(\s|$)', ' ', text)
        words=self.mecab.morphs(words)

        running_offset = 0
        tokens = []
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        return tokens