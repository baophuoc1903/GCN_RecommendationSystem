import numpy as np
import pandas as pd
import os


class Logger:

    def __init__(self, hyps, name="Recall@10"):
        self.hyps = hyps
        self.metric = []
        self.name = name

    def read_log(self, metric):
        self.metric.append(metric)

    def write_log(self):
        path = os.path.join(self.hyps.output_path, f"log_{self.name}_{self.hyps.embedding_size}_{self.hyps.message_dropout}.csv")
        df = pd.DataFrame(columns=["Epoch", f"{self.name}"])
        df["Epoch"] = np.arange(1, len(self.metric)+1)
        df[self.name] = self.metric

        df.to_csv(path, index=False)

