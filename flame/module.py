from abc import ABC, abstractmethod


class Module(ABC):
    def attach(self, frame, module_name):
        self.frame = frame
        self.frame[module_name] = self
        self.module_name = module_name

    @abstractmethod
    def init(self):
        pass
