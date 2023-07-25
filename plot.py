import matplotlib.pyplot as plt


class Plotter:
    plt.rcParams['lines.linewidth']=3
    plt.rcParams['font.size']=14

    def train_curve(self, logger):
        train = logger['train_loss']
        val = logger['val_loss']
        fig, ax = plt.subplots()
        ax.plot(train, label='Training Loss')
        ax.plot(val, label='Validation Loss')
        ax.set_ylim(-.05,1)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MSE')
        ax.legend()
        plt.show()
        return fig
    def scatter_plot(self, true, pred):
        fig, ax = plt.subplots()
        ax.scatter(true, pred)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        plt.show()
        return fig



if __name__ == "__main__":
    import pickle 
    with open('./checkpoints/2023-07-20_test/logs/logs.pkl', 'rb') as f:
        logger = pickle.load(f)
    plotter = Plotter()
    fig = plotter.train_curve(logger)
    fig.savefig('train_cruve_500.png', format='png', dpi=100)
    
