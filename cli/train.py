import logging
import warnings

from argparse import ArgumentParser
from util.logging import logging_config
from util.config import TrainerConfig as Config
# FIXME
from trainer import Trainer
# from trainer import TransTrainer as Trainer
# from trainer import BaseTrainer as Trainer


logger = logging.getLogger(__name__)

    
    
def run(config, mode, use_cpu):
    logger.info("| Mode: {}".format(mode))

    if mode in ["train", "resume"]:
        run_train(config, mode, use_cpu)

    elif mode == "eval":
        run_eval(config, use_cpu)


def run_train(config, mode, use_cpu):
    logger.info("| Training ...")

    logger.info("| Start building model ...")
    trainer = Trainer(config, mode, use_cpu)
    trainer.build()
    
    params = config["modes"][mode]
    total_epoch = params["epoch"]
    eval_epoch = params["eval_epoch"]

    for epoch in range(1, total_epoch + 1):
        logger.info("| Training Epoch: {}/{}".format(epoch, total_epoch))
        trainer.train_run(epoch, total_epoch)

        if epoch % eval_epoch == 0:
            run_eval(config, use_cpu)


def run_eval(config, use_cpu):
    logger.info("| Evaluating ...")
    print(config["modes.eval.domain_datasets.target"])
    for domain_dataset in config["modes.eval.domain_datasets.target"]:
        logger.info("| Start building model ...")
        trainer = Trainer(config, "eval", use_cpu)
        trainer.build(domain_dataset)
        trainer.eval_run()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action="store", type=str, required=True, help="config file path")
    parser.add_argument("--mode", action="store", type=str, choices=["train", "resume", "eval"], required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log", action="store", type=str, required=False, default=None, help="log file path")

    # parse args
    args = parser.parse_args()

    # set up logging config
    logging_config(args.log)

    # init config
    config = Config(config_path=args.config)
    config.load_config()
    config.copy_config()

    # start training model

    run(
        config,
        args.mode,
        args.cpu
    )
