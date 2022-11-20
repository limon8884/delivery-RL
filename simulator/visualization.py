import matplotlib.pyplot as plt


class Trajectory:
    def __init__(self, type) -> None:
        self.points = []
        assert type in ['courier', 'route']
        self.type = type

    def show(self):
        plt.scatter([(p.x, p.y) for p in self.points])
    

class TrajectoryMap:
    def __init__(self) -> None:
        pass

    def show(self):
        pass
