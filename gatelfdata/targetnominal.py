from collections import Counter

class TargetNominal(object):

    def __init__(self,targetStats):
        self.stringCounts = targetStats["stringCounts"]
        self.nrTargets = len(self.stringCounts)
        self.counter = Counter(self.stringCounts)


    def __call__(self, value):
        return value