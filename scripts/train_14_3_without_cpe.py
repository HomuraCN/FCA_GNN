from __future__ import annotations

import argparse

from fca_experiment import add_common_args, add_topk_args, config_from_args, train_bipartite


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 14_3_Bipartite_Weighted_Transformer_without_CPE from generated input.")
    add_common_args(parser, default_experiment="14_3_bipartite_without_cpe")
    add_topk_args(parser)
    parser.set_defaults(hidden_channels=32)
    args = parser.parse_args()

    for topk in args.topk:
        config = config_from_args(args, experiment=args.experiment_name, topk=topk)
        train_bipartite(config, with_cpe=False)


if __name__ == "__main__":
    main()
