import argparse
import pickle

from NeuralPC.utils import get_trainer


def main(args):
    trainer = get_trainer(
        args.trainer,
        model_type=args.model_type,
        optimizer_nm=args.optimizer_nm,
        loss_fn=args.loss_fn,
        data_dir=args.data_dir,
        data=args.data_name,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    trainer.train(args.num_epochs, args.batch_size, args.learning_rate)


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

    args = parser.parse_args()
    main(args)
