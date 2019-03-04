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

# try:
#     import spacy
# except ImportError:
#     spacy = None

logger = logging.getLogger(__name__)


CRF_MODEL_FILE_NAME = "pre_trained_crf_model.pkl"


class PreTrainedTest(EntityExtractor):

    def process(self, message: Message, **kwargs: Any) -> None:

        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Text = None,
             model_metadata: Metadata = None,
             cached_component: Optional['PreTrainedTest'] = None,
             **kwargs: Any
             ) -> 'PreTrainedTest':

        from sklearn.externals import joblib

        #file_name = meta.get("file")
        file_name = "pre_trained_crf_model.pkl"
        model_dir = "models/current/nlu"

        model_file = os.path.join(model_dir, file_name)

        if os.path.exists(model_file):
            ent_tagger = joblib.load(model_file)
            return cls(meta, ent_tagger)
        else:
            return cls(meta)

