import argparse

from NeuralPC.utils import get_trainer
from plot import Plotter


def main(args):
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
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        hidden_dim=args.hidden_dim,
    )

    trainer.train(args.num_epochs, args.batch_size, args.learning_rate)

    # plot training curve
    plotter = Plotter()

    log_path = f"./logs/train_logs/{args.model_type}-{args.data_name}-{args.num_epochs}-B{args.batch_size}-lr{args.learning_rate}/"

    logger = plotter.read_logger(log_path + "model-train.log")
    train_curve = plotter.train_curve(logger, if_log=False)
    train_curve.savefig(
        f"{log_path}/train_curves.pdf",
        format="pdf",
        bbox_inches="tight",
    )


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

    # model specific arguments
    parser.add_argument("--in_dim", type=int, help="Input dimension")
    parser.add_argument("--out_dim", type=int, help="Output dimension")
    parser.add_argument("--hidden_dim", type=int, help="Hidden dimension")

    args = parser.parse_args()
    main(args)
