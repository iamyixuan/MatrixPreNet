from NeuralPC.model import FNN, CNNEncoderDecoder, build_unet
from NeuralPC.utils.data import PreconditionerData, precodition_loss
from NeuralPC.train.trainer import Trainer
from NeuralPC.utils.metrics import test_metrics
from plot import Plotter


def main(args):
    DATANAME = "SPD-ILU-AMG-smoothed_aggregation.npz"
    train_data = PreconditionerData(
        path="./src/data/" + DATANAME,
        mode="train",
    )
    val_data = PreconditionerData(
        path="./src/data/" + DATANAME,
        mode="val",
    )
    test_data = PreconditionerData(
        path="./src/data/" + DATANAME,
        mode="test",
    )
    # net = FNN(in_dim=args.in_dim,
    #           out_dim=args.out_dim,
    #           layer_sizes=args.layer_sizes)
    # net = CNNEncoderDecoder(latent_channels=8)
    net = build_unet()
    trainer = Trainer(
        net=net,
        optimizer_name="Adam",
    )
    if args.train == "True":
        trainer.train(
            train=train_data,
            val=val_data,
            epochs=5000,
            batch_size=128,
            learning_rate=0.001,
            save_freq=50,
            model_name=args.name,
        )
    else:
        true, pred, A = trainer.pred(
            test_data,
            checkpoint="./checkpoints/2023-09-04_AMG-smooth-aggre-solv/model_saved_best",
        )
        import pickle

        with open(
            "./checkpoints/2023-09-04_AMG-smooth-aggre-solv/logs/true-pred-amg.pkl",
            "wb",
        ) as f:
            pickle.dump({"true": true, "pred": pred, "input": A}, f)
        # print(test_metrics(true, pred))

        # plotter = Plotter()
        # fig = plotter.scatter_plot(true, pred)
        # fig.savefig('scatter_500.png',format='png', dpi=100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dim", type=int, default=100 * 100)
    parser.add_argument("-out_dim", type=int, default=100 * 100)
    parser.add_argument(
        "-layer_sizes", type=int, nargs="+", default=[800, 500, 500, 500, 600, 800]
    )
    parser.add_argument("-train", type=str, default="True")
    parser.add_argument("-name", type=str, default="test")

    args = parser.parse_args()

    print(args.layer_sizes)

    main(args)
