import abc


class Container(abc.ABC):
    def __init__(self, input_source):
        self.data = self._prepare_data(input_source)

    def get_data(self):
        return self.data

    def apply(self, editor, inplace=False):
        raise NotImplementedError

    def _prepare_data(self, input_source):
        raise NotImplementedError
