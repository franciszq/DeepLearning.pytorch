from abc import ABCMeta, abstractmethod


class Pipeline(metaclass=ABCMeta):

    @abstractmethod
    def _load_data(self, *args, **kwargs):
        pass

    @abstractmethod
    def _initialize_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_weights(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
