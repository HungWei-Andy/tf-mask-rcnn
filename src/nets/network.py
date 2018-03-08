class Network(object):
    def __init__(self, scope):
        self._is_called = False
        self._scope = scope
        self.input = None
        self.output = None
        self.vars = None
        self.train_vars = None

    def __call__(self, X, *args, **kwargs):
        if self._is_called:
            return self.output
        self.input = X
        self.output = self._build_network(*args, **kwargs)
        self._is_called = True
        return self.output

    def _build_network(self, X, *args, **kwargs):
        raise NotImplementedError

