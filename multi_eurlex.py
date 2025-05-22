# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import datasets


_CITATION = """
@inproceedings{chalkidis-etal-2021-multieurlex,
    title = "{M}ulti{EURLEX} - A multi-lingual and multi-label legal document classification dataset for zero-shot cross-lingual transfer",
    author = "Chalkidis, Ilias  and
      Fergadiotis, Manos  and
      Androutsopoulos, Ion",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.559/",
    pages = "7037--7053"
}
"""

_DESCRIPTION = """
MultiEURLEX is a new multilingual dataset for topic classification of legal documents. 
It contains 65K documents from EUR-LEX, annotated with over 4.3K concepts (labels) from the EuroVoc thesaurus.
The dataset covers 23 official EU languages, plus English whose document coverage is the most comprehensive.
"""

_HOMEPAGE = "https://huggingface.co/datasets/multi_eurlex"

_URL = "https://zenodo.org/record/5532997/files/multi_eurlex.zip"

_EUROVOC_DESCRIPTORS_PATH = "eurovoc_descriptors.json"

_LANGUAGES = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]

_LABEL_LEVELS = ["level_1", "level_2", "level_3"]


class MultiEurlexConfig(datasets.BuilderConfig):
    """BuilderConfig for MultiEURLEX."""

    def __init__(self, language, label_level, languages=None, **kwargs):
        """BuilderConfig for MultiEURLEX.
        
        Args:
            language: Language of the dataset splits.
            label_level: Label level.
            languages: Language list, if `language` is `all_languages`.
            **kwargs: keyword arguments forwarded to super.
        """
        super(MultiEurlexConfig, self).__init__(**kwargs)
        
        self.language = language
        self.label_level = label_level
        
        if language == "all_languages":
            self.languages = languages if languages is not None else _LANGUAGES
        else:
            assert language in _LANGUAGES, "Language must be either `all_languages` or one of the supported languages."


class MultiEurlex(datasets.GeneratorBasedBuilder):
    """MultiEURLEX: A multilingual and multi-label dataset for legal document classification."""

    BUILDER_CONFIGS = []
    
    # Add configurations for individual languages and label levels
    for language in _LANGUAGES + ["all_languages"]:
        for label_level in _LABEL_LEVELS:
            BUILDER_CONFIGS.append(
                MultiEurlexConfig(
                    name=f"{language}-{label_level}",
                    version=datasets.Version("1.0.0"),
                    description=f"MultiEURLEX dataset for language {language} and {label_level} labels.",
                    language=language,
                    label_level=label_level,
                )
            )

    def _info(self):
        # Build dictionary with label-to-index and index-to-label maps
        label_index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_index.json")
        if os.path.exists(label_index_path):
            with open(label_index_path, "r") as fp:
                label_maps = json.load(fp)
        else:
            label_maps = {"label_index": {}, "index_label": {}, "level_1": [], "level_2": [], "level_3": []}
        
        # Build features dictionary
        text_features = {}
        if self.config.language == "all_languages":
            for language in self.config.languages:
                text_features[language] = datasets.Value("string")
        else:
            text_features = datasets.Value("string")
        
        features = {
            "celex_id": datasets.Value("string"),
            "text": text_features,
            "labels": datasets.Sequence(datasets.Value("int32")),
            "concepts": datasets.Sequence(datasets.Value("string"))
        }
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        url = _URL
        data_dir = dl_manager.download_and_extract(url)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.jsonl"),
                    "split": "validation",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        # Load label index
        with open(os.path.join(os.path.dirname(filepath), "label_index.json"), "r", encoding="utf-8") as fp:
            label_maps = json.load(fp)
        
        label_index = label_maps[self.config.label_level]
        
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f):
                item = json.loads(line)
                
                if self.config.language == "all_languages":
                    # Keep only text in languages specified in config
                    text = {lang: item["text"].get(lang) for lang in self.config.languages}
                else:
                    # Extract text for the specific language
                    text = item["text"].get(self.config.language)
                
                # Use only the labels for the specified level
                labels = [label_index.get(concept, -1) for concept in item.get("concepts", [])]
                labels = [label for label in labels if label != -1]  # Filter out labels not in the specified level
                
                example = {
                    "celex_id": item["celex_id"],
                    "text": text,
                    "labels": labels,
                    "concepts": item.get("concepts", [])
                }
                
                yield id_, example
