'''
    Tokenize korean using open-korean-text
       - https://github.com/open-korean-text/open-korean-text
       - http://konlpy.org/ko/latest/api/konlpy.tag/#okt-class
'''

import re

# from konlpy.tag import Okt
from konlpy.tag import Mecab

from rasa_nlu.components import Component
from rasa_nlu.tokenizers import Tokenizer, Token

class KoreanTokenizer(Tokenizer, Component):

    name = "component.KoreanTokenizer"

    provides = ["tokens"]

    '''
    okt implementation
    '''
    # defaults = {
    #     # If True, token normalization
    #     "norm":False,
    #     # If Ture, token stemming
    #     "stem":False
    # }

    def __init__(self, component_config=None):

        super(KoreanTokenizer, self).__init__(component_config)

        self.mecab=Mecab()

        '''
        okt implementation
        '''
        # self.okt=Okt()
        # Load configuration
        # self.norm= self.component_config['norm']
        # self.stem = self.component_config['stem']


    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]

        token_list=self.mecab.morphs(text)

        running_offset = 0
        result = []
        for token in token_list:
            token_offset = text.index(token, running_offset)
            token_len = len(token)
            running_offset = token_offset + token_len
            result.append(Token(token, token_offset))

        return result
