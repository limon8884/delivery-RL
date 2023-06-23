class Timer:
    def __init__(self) -> None:
        self.time = 0
        self.delta_increment = 1

    def Update(self):
        self.time += self.delta_increment

    def __call__(self):
        return self.time
    
    def reset(self):
        self.time = 0
