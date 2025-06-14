from collections import defaultdict
from multiprocessing import cpu_count
from time import time
from typing import List

import numpy as np
from gensim.models import Word2Vec
from gensim.utils import RULE_KEEP


def keep_all_rule(word, count, min_count):
    return RULE_KEEP  # Всегда сохранять слово


class RecWord2Vec:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        self.model = Word2Vec(
            min_count=1,
            window=2,
            vector_size=100,
            sample=6e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            workers=cpu_count() - 1,
            trim_rule=keep_all_rule,
        )

    def fit(self, sessions: List, epochs: int = 17, progress_per: int = 1000):
        sentences = sessions

        if self.verbose:
            word_freq: defaultdict = defaultdict(int)
            for sent in sentences:
                for i in sent:
                    word_freq[i] += 1

            most_freq = ", ".join(sorted(word_freq, reverse=True)[:7])
            msg = "Всего уникальных объявлений: {}, 7 наиболее встречаемых: {}"
            print(msg.format(len(word_freq), most_freq))

        t = time()

        self.model.build_vocab(
            sentences, progress_per=progress_per, trim_rule=keep_all_rule
        )

        if self.verbose:
            print("Время постройки словаря: {} сек".format(round((time() - t), 2)))

        t = time()

        self.model.train(
            sentences,
            total_examples=self.model.corpus_count,
            epochs=epochs,
            report_delay=1,
        )

        if self.verbose:
            print("Время обучения: {} сек".format(round((time() - t), 2)))

        self.model.init_sims(replace=True)

    def predict(self, sessions: np.ndarray, topn: int = 5) -> np.ndarray:
        rec_obj = []
        for session in sessions:
            try:
                pred = list(
                    map(
                        lambda x: x[0],
                        self.model.wv.most_similar(
                            positive=str(session[-1]), topn=topn
                        ),
                    )
                )
            except KeyError:
                print(f"error {session[-1]}")
                pred = ["" for _ in range(topn)]

            rec_obj.append(pred)

        return np.array(rec_obj)
