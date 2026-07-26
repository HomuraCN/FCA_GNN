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
    train_positive_bipartite,
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
        help="Skip adjacency-matrix experiments 12/12_2 and run only bipartite 14_2/14_4/14_3.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run the whole suite once per seed and report mean±std (overrides --seed).",
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    results = []

    for seed in seeds:
        args.seed = seed
        if len(seeds) > 1:
            print(f"===== seed {seed} =====")

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
            config = config_from_args(args, experiment="14_4_bipartite_positive_only_with_cpe", topk=topk)
            config.hidden_channels = args.bipartite_hidden_channels
            results.append(train_positive_bipartite(config, with_cpe=True))

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
            f"{result.experiment}{topk}, seed={result.seed}: "
            f"final_test={result.final_test_acc:.4f}, test@best_val={result.test_acc_at_best_val:.4f}, "
            f"best_test={result.best_test_acc:.4f}@{result.best_test_epoch}"
        )

    if len(seeds) > 1:
        print()
        print("Across seeds (mean±std of final_test / test@best_val):")
        groups: dict[tuple[str, int | None], list] = {}
        for result in results:
            groups.setdefault((result.experiment, result.topk), []).append(result)
        for (experiment, topk), group in groups.items():
            final = [r.final_test_acc for r in group]
            at_best = [r.test_acc_at_best_val for r in group]
            label = experiment if topk is None else f"{experiment}, topK={topk}"
            print(
                f"  {label}: final {mean(final):.4f}±{std(final):.4f}, "
                f"test@best_val {mean(at_best):.4f}±{std(at_best):.4f} (n={len(group)})"
            )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((v - mu) ** 2 for v in values) / (len(values) - 1)) ** 0.5


if __name__ == "__main__":
    main()
