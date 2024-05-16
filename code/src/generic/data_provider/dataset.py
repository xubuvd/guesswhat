class AbstractDataset(object):
    def __init__(self, games):
        self.games = games

    def get_data(self, indices=list()):
        if len(indices) > 0:
            return [self.games[i] for i in indices]
        else:
            return self.games

    def n_examples(self):
        return len(self.games)


class DatasetMerger(AbstractDataset):
    def __init__(self, datasets):
        games = []
        for d in datasets:
            games += d.get_data()
        super(DatasetMerger, self).__init__(games)
