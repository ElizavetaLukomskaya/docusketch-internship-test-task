import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import os

class Drawer:
    def __init__(self):
        pass

    def draw_plots(self, read_file, title=None, xlabel=None, ylabel=None):
        df = pd.DataFrame(read_file)
        X = df[['mean', 'max', 'min', 'floor_mean',
       'floor_max', 'floor_min', 'ceiling_mean', 'ceiling_max', 'ceiling_min']]
        y_real = df['gt_corners']
        y_pred = df['rb_corners']


        plt.scatter(y_pred,  y_pred - y_real,
            c='blue', marker='o', label='Training data')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.legend(loc='upper left')
        plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
        plt.xlim([-10, 50])
        plt.tight_layout()
        plt.savefig('plots/residuals.png')


        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        plt.figure(figsize=(8, 6))
        colors = ['navy', 'turquoise', 'darkorange', 'red']
        lw = 2

        for color, i, target_name in zip(colors, [4, 6, 8, 10], y_real.unique()):
            plt.scatter(X_reduced[y_pred == i, 0], X_reduced[y_pred == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('PCA of dataset')

        plt.show()
        plt.savefig('plots/pca.png')

        relative_path = 'plots'
        path = os.path.abspath(relative_path)

        print(f'You can find all plots by {path}')



        