import argparse
import pickle

from NeuralPC.utils import get_trainer
from plot import Plotter


def main(args):
    model_kwargs = {
        "in_dim": args.in_dim,
        "out_dim": args.out_dim,
        "hidden_dim": args.hidden_dim,
        "in_ch": args.in_ch,
        "out_ch": args.out_ch,
        "kernel_size": args.kernel_size,
    }
    loss_kwargs = {'kind': 'LAL', 'mask': 'True'}
    trainer = get_trainer(
        args.trainer,
        model_type=args.model_type,
        optimizer_nm=args.optimizer_nm,
        loss_fn=args.loss_fn,
        data_dir=args.data_dir,
        data_name=args.data_name,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        additional_info=args.additional_info,
        **model_kwargs,
        **loss_kwargs,
    )

    log_path = f"./experiments/{args.model_type}-{args.data_name}-{args.num_epochs}-B{args.batch_size}-lr{args.learning_rate}-{args.additional_info}/"

    if args.train == "True":

        trainer.train(args.num_epochs, args.batch_size, args.learning_rate)

        # plot training curve
        plotter = Plotter()

        logger = plotter.read_logger(log_path + "model-train.log")
        train_curve = plotter.train_curve(logger, if_log=False)
        train_curve.savefig(
            f"{log_path}/train_curves.pdf",
            format="pdf",
            bbox_inches="tight",
        )

        # save predictions
        val_pred = trainer.predict()
        with open(f"{log_path}/val_pred.pkl", "wb") as f:
            pickle.dump(val_pred, f)
    else:
        if args.model_path is not None:
            val_pred = trainer.predict(args.model_path)
            with open(f"{log_path}/val_pred.pkl", "wb") as f:
                pickle.dump(val_pred, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=str, help="Trainer type")
    parser.add_argument("--model_type", type=str, help="Model type")
    parser.add_argument("--optimizer_nm", type=str, help="Optimizer type")
    parser.add_argument("--loss_fn", type=str, help="Loss function")
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--data_name", type=str, help="dataset name")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--additional_info", type=str, help="Additional info")

    # model specific arguments
    parser.add_argument("--in_dim", type=int, help="Input dimension")
    parser.add_argument("--out_dim", type=int, help="Output dimension")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension")
    parser.add_argument("--in_ch", type=int, help="input channel")
    parser.add_argument("--out_ch", type=int, help="output channel")
    parser.add_argument(
        "--kernel_size", type=int, help="convolution kernel size"
    )

    # train or predict
    parser.add_argument("--train", type=str, help="train or predict")
    parser.add_argument("--model_path", type=str, help="model path")

    args = parser.parse_args()
    main(args)
