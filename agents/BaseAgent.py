from abc import ABC, abstractmethod

class BaseAgent:

    @abstractmethod
    def run(self, query):
        pass


