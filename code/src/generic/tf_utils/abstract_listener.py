class EvaluatorListener(object):
    def __init__(self, require=None):
        self._require = require

    def require(self):
        return self._require

    def valid(self, var):
        return self._require == var

    def before_epoch(self, is_training):
        pass

    def after_epoch(self, is_training):
        pass

    def after_batch(self, result, batch, is_training):
        pass


class EvaluatorListenerAgregator(EvaluatorListener):
    def __init__(self, listeners):
        self.listeners = listeners
        require = [l.require() for l in listeners]
        super(EvaluatorListenerAgregator, self).__init__(require)

    def valid(self, var):
        return var in self._require

    def before_epoch(self, is_training):
        for l in self.listeners:
            l.before_epoch(is_training)

    def after_batch(self, result, batch, is_training):
        for l in self.listeners:
            l.after_batch(result, batch, is_training)