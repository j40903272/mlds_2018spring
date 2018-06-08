from collections import OrderedDict

class History(object):
    def __init__(self):
        super().__init__()
        self.metrics = ['d_loss', 'realAcc', 'g_loss', 'fakeAcc']
        self.history = OrderedDict([(_, self.Average(_)) for _ in self.metrics])
    
    def __iter__(self):
        return self.history.__iter__()
        
    def __getitem__(self, key):
        return self.history[key]
        

    class Average(object):
        def __init__(self, name, num=4):
            super().__init__()
            self.num = num
            self.data = []
            self.name = name

        def value(self):
            if len(self.data) == 0:
                return 0
            return sum(self.data) / len(self.data)

        def append(self, elem):
            self.data.append(elem)
            if self.num is not None and self.num < len(self.data):
                self.data = self.data[-self.num:]

        def extend(self, iters):
            for elem in iters:
                self.append(elem)

        def __repr__(self):
            return '{}: {:3.2f}'.format(self.name, self.value())
            