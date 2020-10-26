import argparse
import datetime
import os

import chainer
import chainer.links as L

from chainer.iterators import MultiprocessIterator
from chainer.optimizers import Adam
from chainer.training import StandardUpdater, Trainer, extensions
from tensorboardX import SummaryWriter

from config.config import parse_config
from datasets.image_dataset import BaseImageDataset
from networks.handwriting import HandwritingNet
from utils.backup import get_import_info
from utils.logger import Logger
from utils.tensorboard.tensorboard_evaluator import TensorboardEvaluator
from utils.tensorboard.tensorboard_gradient_histogram import TensorboardGradientPlotter


def prepare_log_dir(args):
    args.log_dir = os.path.join("logs", args.log_dir, "{}_{}".format(datetime.datetime.now().isoformat(), args.log_name))
    os.makedirs(args.log_dir, exist_ok=True)
    return args


def main(args):
    args = prepare_log_dir(args)

    # set dtype for training
    chainer.global_config.dtype = args.dtype

    train_dataset = BaseImageDataset(
        args.train_file,
        args.image_size,
        root=os.path.dirname(args.train_file),
        dtype=chainer.get_dtype(),
    )

    validation_dataset = BaseImageDataset(
        args.val_file,
        args.image_size,
        root=os.path.dirname(args.val_file),
        dtype=chainer.get_dtype(),
    )

    train_iter = MultiprocessIterator(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_iter = MultiprocessIterator(validation_dataset, batch_size=args.batch_size, repeat=False)

    net = HandwritingNet()
    model = L.Classifier(net, label_key='has_text')

    tensorboard_handle = SummaryWriter(log_dir=args.log_dir)

    optimizer = Adam(alpha=args.learning_rate)
    optimizer.setup(model)
    if args.save_gradient_information:
        optimizer.add_hook(
            TensorboardGradientPlotter(tensorboard_handle, args.log_interval),
        )

    # log train information everytime we encouter a new epoch or args.log_interval iterations have been done
    log_interval_trigger = (
        lambda trainer:
        (trainer.updater.is_new_epoch or trainer.updater.iteration % args.log_interval == 0)
        and trainer.updater.iteration > 0
    )

    updater = StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = Trainer(updater, (args.num_epoch, 'epoch'), out=args.log_dir)

    data_to_log = {
        'log_dir': args.log_dir,
        'image_size': args.image_size,
        # 'num_layers': args.num_layers,
        'keep_aspect_ratio': train_dataset.keep_aspect_ratio,
        'net': get_import_info(net),
    }

    for argument in filter(lambda x: not x.startswith('_'), dir(args)):
        data_to_log[argument] = getattr(args, argument)

    def backup_train_config(stats_cpu):
        iteration = stats_cpu.pop('iteration')
        epoch = stats_cpu.pop('epoch')
        elapsed_time = stats_cpu.pop('elapsed_time')

        for key, value in stats_cpu.items():
            tensorboard_handle.add_scalar(key, value, iteration)

        if iteration == args.log_interval:
            stats_cpu.update(data_to_log)

        stats_cpu.update({
            "epoch": epoch,
            "iteration": iteration,
            "elapsed_time": elapsed_time,
        })

    trainer.extend(
        extensions.snapshot_object(net, net.__class__.__name__ + '_{.updater.iteration}.npz'),
        trigger=lambda trainer: trainer.updater.is_new_epoch or trainer.updater.iteration % args.snapshot_interval == 0,
    )

    trainer.extend(
        extensions.snapshot(filename='trainer_snapshot', autoload=args.resume is not None),
        trigger=(args.snapshot_interval, 'iteration')
    )

    trainer.extend(
        TensorboardEvaluator(
            validation_iter,
            model,
            device=args.gpu,
            tensorboard_handle=tensorboard_handle
        ),
        trigger=(args.test_interval, 'iteration'),
    )

    logger = Logger(
        os.path.dirname(os.path.realpath(__file__)),
        args.log_dir,
        postprocess=backup_train_config,
        trigger=log_interval_trigger,
        exclusion_filters=['*logs*', '*.pyc', '__pycache__', '.git*'],
        resume=args.resume is not None,
    )

    trainer.extend(logger, trigger=log_interval_trigger)
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'main/accuracy', 'validation/main/accuracy'],
            log_report='Logger',
        ),
        trigger=log_interval_trigger,
    )

    trainer.extend(extensions.ExponentialShift('alpha', 0.1, optimizer=optimizer), trigger=(10, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model that can do handwriting localization/determination on WPI data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("log_name", help="name of the log dir to create")
    parser.add_argument("config", help="path to config file to use for training")
    parser.add_argument("--log-dir", default="test", help="name of the subdir where we want to put logs")
    parser.add_argument("--gpu", default="-1", help="gpu or device to use, ints > 0 indicate a GPU, -1 == CPU, everything else is chainerx")
    parser.add_argument("--resume", help="path to log dir from where to resume training from")
    parser.add_argument("--save-gradient", action='store_true', dest="save_gradient_information", default=False, help="plot gradient information to tensorbard")

    args = parser.parse_args()
    args = parse_config(args.config, args)
    main(args)
