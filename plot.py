import matplotlib.pyplot as plt
import pandas as pd

def plot():
    statistics_df = pd.read_csv('statistics/negative_critic_losses.csv', sep=' ', index_col='epoch')
    ax = statistics_df.plot()
    ax.set_ylabel("Negative critic loss")
    ax.set_xlabel("Epoch")
    plt.savefig('visualizations/loss_curves.png')