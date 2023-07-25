from src.model import FNN
from src.utils.data import PreconditionerData
from src.train.trainer import Trainer
from src.utils.metrics import test_metrics
from plot import Plotter



def main(args):
    train_data = PreconditionerData(path='./src/data/training_data_1000.npz',
                             mode='train',
                             )
    val_data = PreconditionerData(path='./src/data/training_data_1000.npz',
                             mode='val',
                             )
    test_data = PreconditionerData(path='./src/data/training_data_1000.npz',
                             mode='test',
                             )
    net = FNN(in_dim=args.in_dim,
              out_dim=args.out_dim,
              layer_sizes=args.layer_sizes)

    trainer = Trainer(net=net,
                             optimizer_name='Adam',
                             loss_name='MSE',
                             )
    if args.train == "True":
        trainer.train(train=train_data,
                    val=val_data,
                    epochs=2000,
                    batch_size=128,
                    learning_rate=0.01,
                    save_freq=50,
                    model_name=args.name
        )
    else:
        true, pred = trainer.pred(test_data, checkpoint='./checkpoints/2023-07-20_test/model_saved_best')
        print(test_metrics(true, pred))
    
        plotter = Plotter()
        fig = plotter.scatter_plot(true, pred)
        fig.savefig('scatter_500.png',format='png', dpi=100)
    

if __name__ == "__main__":
    import argparse
    

    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dim', type=int, default=1000*1000)
    parser.add_argument('-out_dim', type=int, default=1000)
    parser.add_argument('-layer_sizes', type=int, nargs='+', default=[512, 256, 128])
    parser.add_argument('-train', type=str, default='True')
    parser.add_argument('-name', type=str, default='test')

    args = parser.parse_args()

    print(args.layer_sizes)

    main(args)


