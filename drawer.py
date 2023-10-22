import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PlotDrawer:
    def __init__(self):
        pass

    def draw_plots(self, read_file, title=None, xlabel=None, ylabel=None):

        df = pd.DataFrame(read_file)