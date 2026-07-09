from __future__ import annotations

import argparse

from fca_experiment import add_adjacency_args, add_common_args, config_from_args, train_adjacency


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 12_2_Transformer_without_CPE from a generated experiment input directory.")
    add_common_args(parser, default_experiment="12_without_cpe")
    add_adjacency_args(parser)
    args = parser.parse_args()

    config = config_from_args(args, experiment=args.experiment_name)
    train_adjacency(config, with_cpe=False)


if __name__ == "__main__":
    main()
