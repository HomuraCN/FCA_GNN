from __future__ import annotations

import argparse

from fca_experiment import (
    add_adjacency_args,
    add_common_args,
    add_cpe_args,
    add_topk_args,
    config_from_args,
    train_adjacency,
    train_bipartite,
    write_suite_summary,
)


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the core FCA-GNN experiment suite.")
    add_common_args(parser, default_experiment="suite")
    add_adjacency_args(parser)
    add_cpe_args(parser)
    add_topk_args(parser)
    parser.add_argument("--bipartite-hidden-channels", type=int, default=32)
    parser.add_argument(
        "--skip-adjacency",
        "--skip",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Skip adjacency-matrix experiments 12/12_2 and run only bipartite 14_2/14_3.",
    )
    args = parser.parse_args()

    results = []

    if args.skip_adjacency:
        print("Skipping adjacency-matrix experiments: 12_with_cpe and 12_without_cpe")
    else:
        config = config_from_args(args, experiment="12_with_cpe")
        results.append(train_adjacency(config, with_cpe=True))

        config = config_from_args(args, experiment="12_without_cpe")
        results.append(train_adjacency(config, with_cpe=False))

    for topk in args.topk:
        config = config_from_args(args, experiment="14_2_bipartite_with_cpe", topk=topk)
        config.hidden_channels = args.bipartite_hidden_channels
        results.append(train_bipartite(config, with_cpe=True))

    for topk in args.topk:
        config = config_from_args(args, experiment="14_3_bipartite_without_cpe", topk=topk)
        config.hidden_channels = args.bipartite_hidden_channels
        results.append(train_bipartite(config, with_cpe=False))

    summary_path = write_suite_summary(args.output_dir, args.dataset, results)
    print()
    print(f"Suite summary: {summary_path}")
    for result in results:
        topk = "" if result.topk is None else f", topK={result.topk}"
        print(
            f"{result.experiment}{topk}: "
            f"final_test={result.final_test_acc:.4f}, best_test={result.best_test_acc:.4f}@{result.best_test_epoch}"
        )


if __name__ == "__main__":
    main()
