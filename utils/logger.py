import matplotlib.pyplot as plt
import numpy as np
import torch


class Logger:

    def __init__(self, hyps):
        self.hyps = hyps
        self.metric = []

    def read_log(self, loss, metric):
        self.metric.append(metric)

    def visualize(self):
        pass

