
class State:
    def __init__(self) -> None:
        pass

class Environment(object):
    def __init__(self) -> None:
        self.state = State()

    def reset(self):
        pass

    def step(self, action):
        '''
        returns: reward
        '''
        pass

    def get_state(self):
        '''
        returns the copy of state via inherence
        '''
        pass
