import matplotlib.pyplot as plt
import seaborn as sns


class Trajectory:
    def __init__(self, type, id=0) -> None:
        self.points = []
        assert type in ['courier', 'route']
        self.type = type
        self.id = id

    def show(self):
        plt.scatter([(p.x, p.y) for p in self.points])

    

class TrajectoryMap:
    def __init__(self) -> None:
        pass

    def show(self):
        pass
