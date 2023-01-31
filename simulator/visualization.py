import matplotlib.pyplot as plt



class Visualizer:
    def __init__(self, simulator) -> None:
        self.simulator = simulator
        self.fig, self.ax = plt.subplots()
        self.fig.set_figheight(8)
        self.fig.set_figwidth(8)
    
    def DrawFrame(self):
        self.simulator.corner_bounds[0].plot(self.ax, color='black')
        self.simulator.corner_bounds[1].plot(self.ax, color='black')

    def Update(self):
        for ar in self.simulator.active_routes:
            ar.plot(self.ax)

        return self.ax

    def Show(self):
        return self.fig.show()

