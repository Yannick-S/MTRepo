class lr_exp:
    def __init__(self, start=1, decay=0.9):
        self.start = start
        self.decay = decay

        self.current = start

    def next(self):
        self.current = self.current * self.decay

        return self.current