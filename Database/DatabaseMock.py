import pandas as pd


class DatabaseMock:
    def __init__(self, dataset_url):
        self._data = pd.read_csv(dataset_url)