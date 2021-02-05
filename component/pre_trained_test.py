from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
import os
from konlpy.tag import Mecab
import eli5
import typing
from builtins import str
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa_nlu.config import RasaNLUModelConfig, InvalidConfigError
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from component.ner_crf import KoreanExtractor

# try:
#     import spacy
# except ImportError:
#     spacy = None

logger = logging.getLogger(__name__)
mecab = Mecab()

CRF_MODEL_FILE_NAME = "pre_trained_crf_model.pkl"


class PreTrainedCRF(EntityExtractor):

    provides = ["entities"]

    requires = ["tokens"]

    defaults = {
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        "BILOU_flag": False,

        # crf_features is [before, word, after] array with before, word,
        # after holding keys about which
        # features to use for each word, for example, 'title' in
        # array before will have the feature
        # "is the preceding word in title case?"
        # POS features require spaCy to be installed
        "features": [
            ["low", "length", "kor_pos", "kor_pos_2", "prefix1"],
            ["kor_pos", "kor_pos_2", "length", "bias", "low", "prefix1", "suffix1", "digit", "pattern"],
            ["low", "length", "kor_pos", "kor_pos_2", "prefix1"]
        ],

        # The maximum number of iterations for optimization algorithms.
        "max_iterations": 50,

        # weight of theL1 regularization
        "L1_c": 0.1,

        # weight of the L2 regularization
        "L2_c": 0.1
    }

    function_dict = {
        'low': lambda doc: doc[0].lower(),
        'prefix1': lambda doc: doc[0][:1],
        'prefix2': lambda doc: doc[0][:2],
        'prefix3': lambda doc: doc[0][:3],
        'suffix3': lambda doc: doc[0][-3:],
        'suffix2': lambda doc: doc[0][-2:],
        'suffix1': lambda doc: doc[0][-1:],
        #'pos': lambda doc: doc[1],
        #'pos2': lambda doc: doc[1][:2],
        'bias': lambda doc: 'bias',
        'length': lambda doc: len(doc[0]),
        'kor_pos': lambda doc: mecab.pos(doc[0])[0][1],
        'kor_pos_2': lambda doc: mecab.pos(doc[0])[0][1][:2],
        'digit': lambda doc: doc[0].isdigit(),
        'pattern': lambda doc: doc[3],
    }

    def __init__(self, component_config=None, ent_tagger=None):
        # type: (sklearn_crfsuite.CRF, Dict[Text, Any]) -> None

        super(PreTrainedCRF, self).__init__(component_config)

        self.ent_tagger = ent_tagger

        self.pos_features = False

        self.mecab = Mecab()

        self.crf = KoreanExtractor()

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[PreTrainedCRF]
             **kwargs  # type: **Any
             ):
        # type: (...) -> PreTrainedCRF
        from sklearn.externals import joblib

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("classifier_file", CRF_MODEL_FILE_NAME)
        model_file = os.path.join(model_dir, file_name)

        if os.path.exists(model_file):
            ent_tagger = joblib.load(model_file)
            return cls(meta, ent_tagger)
        else:
            return cls(meta)

    def process(self, message: Message, **kwargs: Any) -> None:

        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    def extract_entities(self, message):
        # type: (Message) -> List[Dict[Text, Any]]
        """Take a sentence and return entities in json format"""

        if self.ent_tagger is not None:
            text_data = self._from_text_to_crf(message)
            features = self._sentence_to_features(text_data)
            ents = self.ent_tagger.predict_marginals_single(features)
            return self._from_crf_to_json(message, ents)
        else:
            return []

    def _from_text_to_crf(self, message, entities=None):
        # type: (Message, List[Text]) -> List[Tuple[Text, Text, Text, Text]]
        """Takes a sentence and switches it to crfsuite format."""

        crf_format = []
        if self.pos_features:
            tokens = message.get("spacy_doc")
        else:
            tokens = message.get("tokens")
        for i, token in enumerate(tokens):
            pattern = self.__pattern_of_token(message, i)
            entity = entities[i] if entities else "N/A"
            """korean_pos_tagging"""
            tag = mecab.pos(token.text)[0][1]
            # tag = self.__tag_of_token(token) if self.pos_features else None
            crf_format.append((token.text, tag, entity, pattern))
        return crf_format

    @staticmethod
    def __pattern_of_token(message, i):
        if message.get("tokens") is not None:
            return message.get("tokens")[i].get("pattern", {})
        else:
            return {}

    def _from_crf_to_json(self, message, entities):
        # type: (Message, List[Any]) -> List[Dict[Text, Any]]

        # if self.pos_features:
        #     tokens = message.get("spacy_doc")
        # else:
        tokens = message.get("tokens")

        if len(tokens) != len(entities):
            raise Exception('Inconsistency in amount of tokens '
                            'between crfsuite and message')

        if self.component_config["BILOU_flag"]:
            return self._convert_bilou_tagging_to_entity_result(
                tokens, entities)
        else:
            # not using BILOU tagging scheme, multi-word entities are split.
            return self._convert_simple_tagging_to_entity_result(
                tokens, entities)

    def _convert_simple_tagging_to_entity_result(self, tokens, entities):
        json_ents = []

        for word_idx in range(len(tokens)):
            entity_label, confidence = self.most_likely_entity(
                    word_idx, entities)
            word = tokens[word_idx]
            if entity_label != 'O':
                if self.pos_features:
                    start = word.idx
                    end = word.idx + len(word)
                else:
                    start = word.offset
                    end = word.end
                ent = {'start': start,
                       'end': end,
                       'value': word.text,
                       'entity': entity_label,
                       'confidence': confidence}
                json_ents.append(ent)

        return json_ents

    def most_likely_entity(self, idx, entities):
        if len(entities) > idx:
            entity_probs = entities[idx]
        else:
            entity_probs = None
        if entity_probs:
            label = max(entity_probs,
                        key=lambda key: entity_probs[key])
            if self.component_config["BILOU_flag"]:
                # if we are using bilou flags, we will combine the prob
                # of the B, I, L and U tags for an entity (so if we have a
                # score of 60% for `B-address` and 40% and 30%
                # for `I-address`, we will return 70%)
                return label, sum([v
                                   for k, v in entity_probs.items()
                                   if k[2:] == label[2:]])
            else:
                return label, entity_probs[label]
        else:
            return "", 0.0

    def _sentence_to_features(self, sentence):
        # type: (List[Tuple[Text, Text, Text, Text]]) -> List[Dict[Text, Any]]
        """Convert a word into discrete features in self.crf_features,
        including word before and word after."""

        configured_features = self.component_config["features"]
        sentence_features = []

        for word_idx in range(len(sentence)):
            # word before(-1), current word(0), next word(+1)
            feature_span = len(configured_features)
            half_span = feature_span // 2
            feature_range = range(- half_span, half_span + 1)
            prefixes = [str(i) for i in feature_range]
            word_features = {}
            for f_i in feature_range:
                if word_idx + f_i >= len(sentence):
                    word_features['EOS'] = True
                    # End Of Sentence
                elif word_idx + f_i < 0:
                    word_features['BOS'] = True
                    # Beginning Of Sentence
                else:
                    word = sentence[word_idx + f_i]
                    f_i_from_zero = f_i + half_span
                    prefix = prefixes[f_i_from_zero]
                    features = configured_features[f_i_from_zero]
                    for feature in features:
                        if feature == "pattern":
                            # add all regexes as a feature
                            regex_patterns = self.function_dict[feature](word)
                            for p_name, matched in regex_patterns.items():
                                feature_name = prefix + ":" + feature + ":" + p_name
                                word_features[feature_name] = matched
                        else:
                            # append each feature to a feature vector
                            value = self.function_dict[feature](word)
                            word_features[prefix + ":" + feature] = value
            sentence_features.append(word_features)
        return sentence_features
