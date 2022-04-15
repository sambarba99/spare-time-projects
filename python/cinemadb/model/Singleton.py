# Helper class for implementing singletons (to be used as a decorator)
# Author: Sam Barba
# Created 15/04/2022

class Singleton:
    def __init__(self, decorated_class):
        self.instance = None
        self.decorated_class = decorated_class

    def get_instance(self):
        if self.instance is None:
            self.instance = self.decorated_class()
        return self.instance

    def __call__(self):
        raise TypeError("Singletons must be accessed via get_instance()")
