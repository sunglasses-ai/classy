from argparse import ArgumentParser
from pathlib import Path
from hydra import compose, initialize
from classy.scripts.model.train import train, fix


def parse_args():
    parser = ArgumentParser()

    # TODO: add help?
    parser.add_argument("task", choices=("sequence", "token", "sentence-pair"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("-m", "--model-name", default="bert")
    parser.add_argument("-n", "--exp-name", "--experiment-name", dest="exp_name", default=None)
    parser.add_argument("-d", "--device", default="gpu")  # TODO: add validator?
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("-c", "--config", nargs="+", default=[])

    return parser.parse_args()


def main():
    args = parse_args()

    if args.root is not None:
        config_name = args.root
    else:
        task = "sent-pair" if args.task == "sentence-pair" else args.task
        config_name = f"{task}-{args.model_name}"

    overrides = []

    # choose device
    device = "cuda" if args.device == "gpu" else args.device
    overrides.append(f"device={device}")

    # create default experiment name if not provided
    exp_name = args.exp_name or f"{args.task}-{args.model_name}"
    overrides.append(f"exp_name={exp_name}")

    overrides.append(f"data.datamodule.dataset_path={args.dataset}")
    # overrides.append(f"datamodule.task={args.task}")

    # append all user-provided configuration overrides
    overrides += args.config

    from omegaconf import OmegaConf

    initialize(config_path="../../../configurations/")
    conf = compose(config_name=config_name, overrides=overrides)
    print(OmegaConf.to_yaml(conf, resolve=True))

    # cannot call main() here because it's wrapped by hydra (...right? TODO: check)
    fix(conf)
    train(conf)


def test(cmd):
    import sys

    sys.argv = cmd.split(" ")
    print(cmd, end=" -> \n\t")
    main()


if __name__ == "__main__":
    main()
    # test("train.py sentence-pair data/glue/mrpc")
    # test("train.py token data/mrpc -m small -n mrpc-small")
    # test(
    #     "train.py token data/mrpc -m small "
    #     "-c training.pl_trainer.val_check_interval=1.0 data.pl_module.batch_size=16"
    # )
    # test("train.py sentence data/s")
