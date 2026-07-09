from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dense_to_sparse


@dataclass
class TrainConfig:
    dataset: str
    input_dir: Path
    output_dir: Path
    experiment: str
    hidden_channels: int
    heads: int
    learning_rate: float
    weight_decay: float
    epochs: int
    dropout: float
    seed: int
    log_every: int
    cpe_profile_bins: int = 8
    threshold_pos: float = 128.0
    threshold_neg: float = 5000.0
    topk: int | None = None


@dataclass
class TrainResult:
    dataset: str
    experiment: str
    topk: int | None
    final_train_acc: float
    final_val_acc: float
    final_test_acc: float
    best_val_acc: float
    best_val_epoch: int
    best_test_acc: float
    best_test_epoch: int
    log_dir: str
    epoch_metrics_path: str
    result_json_path: str


def require_absolute_path(value: str, argument: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"{argument} must be an absolute path: {value}")
    return path


def add_common_args(parser: argparse.ArgumentParser, *, default_experiment: str) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-dir", required=True, type=lambda v: require_absolute_path(v, "--input-dir"))
    parser.add_argument("--output-dir", required=True, type=lambda v: require_absolute_path(v, "--output-dir"))
    parser.add_argument("--experiment-name", default=default_experiment)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)


def add_cpe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cpe-profile-bins", type=int, default=8)


def add_adjacency_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threshold-pos", type=float, default=128.0)
    parser.add_argument("--threshold-neg", type=float, default=5000.0)


def add_topk_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=[8, 32],
        help="Run one or more per-object membership topK values.",
    )


def config_from_args(args: argparse.Namespace, *, experiment: str, topk: int | None = None) -> TrainConfig:
    return TrainConfig(
        dataset=args.dataset,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        experiment=experiment,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        dropout=args.dropout,
        seed=args.seed,
        log_every=max(1, args.log_every),
        cpe_profile_bins=getattr(args, "cpe_profile_bins", 8),
        threshold_pos=getattr(args, "threshold_pos", 128.0),
        threshold_neg=getattr(args, "threshold_neg", 5000.0),
        topk=topk,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_root(config: TrainConfig) -> Path:
    root = config.output_dir / f"{config.dataset}_output"
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    return root


def make_run_dir(config: TrainConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"_topk{config.topk}" if config.topk is not None else ""
    run_dir = experiment_root(config) / "runs" / f"{config.dataset}_{config.experiment}{suffix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_feature_matrix(path: Path) -> torch.Tensor:
    values = np.loadtxt(path, delimiter=",")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return torch.tensor(values, dtype=torch.float)


def load_labels(input_dir: Path, dataset: str, expected_num_nodes: int) -> np.ndarray:
    label_path = input_dir / f"{dataset}.labels.csv"
    if label_path.exists():
        labels = pd.read_csv(label_path, header=None).iloc[:, 0].to_numpy()
        if len(labels) != expected_num_nodes:
            raise ValueError(f"Label count mismatch: expected {expected_num_nodes}, got {len(labels)} at {label_path}")
        return labels

    candidates = [input_dir / f"{dataset}.data", input_dir / f"{dataset}.data.csv", input_dir / f"{dataset}.csv"]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path, header=None).dropna(axis=1, how="all")
        if len(df) == expected_num_nodes:
            return df.iloc[:, -1].to_numpy()
    raise ValueError(f"Cannot find labels for {expected_num_nodes} objects in {input_dir}")


def normalize_edge_attr(edge_attr: torch.Tensor) -> torch.Tensor:
    edge_attr = torch.log1p(edge_attr.float())
    if edge_attr.numel() == 0:
        return edge_attr
    min_value = edge_attr.min()
    max_value = edge_attr.max()
    if max_value > min_value:
        edge_attr = (edge_attr - min_value) / (max_value - min_value)
    return edge_attr


def make_masks(num_nodes: int, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    num_train = int(num_nodes * 0.6)
    num_val = int(num_nodes * 0.2)
    indices = torch.randperm(num_nodes) if generator is None else torch.randperm(num_nodes, generator=generator)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train : num_train + num_val]] = True
    test_mask[indices[num_train + num_val :]] = True
    return train_mask, val_mask, test_mask


def load_depth_profile_cpe(input_dir: Path, dataset: str, profile_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    cpe_pos = load_feature_matrix(input_dir / f"{dataset}_CPE_A_plus_depth_profile{profile_bins}.csv")
    cpe_neg = load_feature_matrix(input_dir / f"{dataset}_CPE_A_negative_depth_profile{profile_bins}.csv")
    return cpe_pos, cpe_neg


def load_adjacency_data(config: TrainConfig, *, with_cpe: bool) -> tuple[Data, int]:
    x_features = load_feature_matrix(config.input_dir / f"{config.dataset}.data.cleaned.csv")
    num_nodes = x_features.shape[0]

    a_pos = torch.tensor(np.loadtxt(config.input_dir / f"{config.dataset}_A_plus_UG.csv", delimiter=","), dtype=torch.float)
    if a_pos.shape != (num_nodes, num_nodes):
        raise ValueError(f"Positive adjacency shape should be {(num_nodes, num_nodes)}, got {a_pos.shape}")
    a_pos[a_pos <= config.threshold_pos] = 0
    a_pos.fill_diagonal_(0)
    edge_index_pos, edge_attr_pos = dense_to_sparse(a_pos)
    edge_attr_pos = normalize_edge_attr(edge_attr_pos)

    a_neg = torch.tensor(np.loadtxt(config.input_dir / f"{config.dataset}_A_negative_UG.csv", delimiter=","), dtype=torch.float)
    if a_neg.shape != (num_nodes, num_nodes):
        raise ValueError(f"Negative adjacency shape should be {(num_nodes, num_nodes)}, got {a_neg.shape}")
    a_neg[a_neg <= config.threshold_neg] = 0
    a_neg.fill_diagonal_(0)
    edge_index_neg, edge_attr_neg = dense_to_sparse(a_neg)
    edge_attr_neg = normalize_edge_attr(edge_attr_neg)

    if with_cpe:
        cpe_pos, cpe_neg = load_depth_profile_cpe(config.input_dir, config.dataset, config.cpe_profile_bins)
        if cpe_pos.shape[0] != num_nodes or cpe_neg.shape[0] != num_nodes:
            raise ValueError(f"CPE row count must match object count {num_nodes}")
        x_pos = torch.cat([x_features, cpe_pos], dim=1)
        x_neg = torch.cat([x_features, cpe_neg], dim=1)
    else:
        x_pos = x_features
        x_neg = x_features

    labels = load_labels(config.input_dir, config.dataset, num_nodes)
    encoder = LabelEncoder()
    y_numpy = encoder.fit_transform(labels)
    y = torch.tensor(y_numpy, dtype=torch.long)
    train_mask, val_mask, test_mask = make_masks(num_nodes)

    data = Data(
        x_pos=x_pos,
        x_neg=x_neg,
        y=y,
        edge_index_pos=edge_index_pos,
        edge_attr_pos=edge_attr_pos.view(-1, 1),
        edge_index_neg=edge_index_neg,
        edge_attr_neg=edge_attr_neg.view(-1, 1),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )
    return data, len(np.unique(y_numpy))


class DualConceptTransformer(nn.Module):
    def __init__(self, pos_in_channels: int, neg_in_channels: int, hidden_channels: int, out_channels: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.pos_conv = TransformerConv(pos_in_channels, hidden_channels, heads=heads, edge_dim=1)
        self.neg_conv = TransformerConv(neg_in_channels, hidden_channels, heads=heads, edge_dim=1)
        self.fusion_layer = nn.Linear(hidden_channels * heads * 2, out_channels)

    def forward(self, x_pos, x_neg, edge_index_pos, edge_attr_pos, edge_index_neg, edge_attr_neg):
        h_pos = self.pos_conv(x_pos, edge_index_pos, edge_attr_pos)
        h_pos = F.dropout(F.relu(h_pos), p=self.dropout, training=self.training)
        h_neg = self.neg_conv(x_neg, edge_index_neg, edge_attr_neg)
        h_neg = F.dropout(F.relu(h_neg), p=self.dropout, training=self.training)
        return self.fusion_layer(torch.cat([h_pos, h_neg], dim=1))


def train_adjacency(config: TrainConfig, *, with_cpe: bool) -> TrainResult:
    set_seed(config.seed)
    data, num_classes = load_adjacency_data(config, with_cpe=with_cpe)
    model = DualConceptTransformer(
        pos_in_channels=data.x_pos.shape[1],
        neg_in_channels=data.x_neg.shape[1],
        hidden_channels=config.hidden_channels,
        out_channels=num_classes,
        heads=config.heads,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    return run_training_loop(
        config,
        model,
        optimizer,
        criterion,
        forward_fn=lambda: model(data.x_pos, data.x_neg, data.edge_index_pos, data.edge_attr_pos, data.edge_index_neg, data.edge_attr_neg),
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )


def keep_topk_memberships_per_object(df: pd.DataFrame, topk: int | None) -> pd.DataFrame:
    if topk is None or topk <= 0 or len(df) == 0:
        return df
    return (
        df.sort_values(["object_id", "weight", "concept_id"], ascending=[True, False, True])
        .groupby("object_id", group_keys=False)
        .head(topk)
        .reset_index(drop=True)
    )


def load_bipartite_edges(path: Path, object_count: int, concept_count: int, topk_per_object: int | None) -> dict:
    if path.exists():
        df = pd.read_csv(path)
    elif Path(str(path) + ".gz").exists():
        df = pd.read_csv(Path(str(path) + ".gz"), compression="gzip")
    else:
        raise FileNotFoundError(path)

    required_columns = {"object_id", "concept_id", "weight"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Edge table must contain columns {required_columns}: {path}")

    original_edge_count = len(df)
    df = keep_topk_memberships_per_object(df, topk_per_object)
    object_ids = torch.tensor(df["object_id"].to_numpy(), dtype=torch.long)
    concept_ids = torch.tensor(df["concept_id"].to_numpy(), dtype=torch.long)
    weights = torch.tensor(df["weight"].to_numpy(), dtype=torch.float).view(-1, 1)

    if object_ids.numel() > 0:
        if object_ids.min() < 0 or object_ids.max() >= object_count:
            raise ValueError(f"Object id out of range: {path}")
        if concept_ids.min() < 0 or concept_ids.max() >= concept_count:
            raise ValueError(f"Concept id out of range: {path}")

    return {
        "obj_to_concept": torch.stack([object_ids, concept_ids], dim=0),
        "concept_to_obj": torch.stack([concept_ids, object_ids], dim=0),
        "edge_attr": weights,
        "rev_edge_attr": weights.clone(),
        "original_edge_count": original_edge_count,
        "kept_edge_count": len(df),
    }


def load_bipartite_batch(config: TrainConfig, *, with_cpe: bool) -> dict:
    x_raw = load_feature_matrix(config.input_dir / f"{config.dataset}.data.cleaned.csv")
    num_objects = x_raw.shape[0]
    if with_cpe:
        cpe_pos, cpe_neg = load_depth_profile_cpe(config.input_dir, config.dataset, config.cpe_profile_bins)
        if cpe_pos.shape[0] != num_objects or cpe_neg.shape[0] != num_objects:
            raise ValueError(f"CPE row count must match object count {num_objects}")
        x_pos = torch.cat([x_raw, cpe_pos], dim=1)
        x_neg = torch.cat([x_raw, cpe_neg], dim=1)
    else:
        x_pos = x_raw
        x_neg = x_raw

    pos_concept_x = load_feature_matrix(config.input_dir / f"{config.dataset}_positive_object_concept_concept_features.csv")
    neg_concept_x = load_feature_matrix(config.input_dir / f"{config.dataset}_negative_object_concept_concept_features.csv")
    pos_edges = load_bipartite_edges(
        config.input_dir / f"{config.dataset}_positive_object_concept_edges.csv",
        num_objects,
        pos_concept_x.shape[0],
        config.topk,
    )
    neg_edges = load_bipartite_edges(
        config.input_dir / f"{config.dataset}_negative_object_concept_edges.csv",
        num_objects,
        neg_concept_x.shape[0],
        config.topk,
    )

    labels = load_labels(config.input_dir, config.dataset, num_objects)
    encoder = LabelEncoder()
    y_numpy = encoder.fit_transform(labels)
    y = torch.tensor(y_numpy, dtype=torch.long)
    train_mask, val_mask, test_mask = make_masks(num_objects, config.seed)

    print(f"topK={config.topk}")
    print(f"object feature dim: {x_raw.shape[1]}")
    print(f"positive concept nodes: {pos_concept_x.shape[0]}, edges: {pos_edges['kept_edge_count']}/{pos_edges['original_edge_count']}")
    print(f"negative concept nodes: {neg_concept_x.shape[0]}, edges: {neg_edges['kept_edge_count']}/{neg_edges['original_edge_count']}")

    return {
        "x_pos": x_pos,
        "x_neg": x_neg,
        "pos_concept_x": pos_concept_x,
        "neg_concept_x": neg_concept_x,
        "pos_edges": pos_edges,
        "neg_edges": neg_edges,
        "y": y,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "num_classes": len(np.unique(y_numpy)),
    }


class WeightedBipartiteBranch(nn.Module):
    def __init__(self, object_in_channels: int, concept_in_channels: int, hidden_channels: int, heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.object_encoder = nn.Linear(object_in_channels, hidden_channels)
        self.concept_encoder = nn.Linear(concept_in_channels, hidden_channels)
        self.object_to_concept = TransformerConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, concat=False)
        self.concept_to_object = TransformerConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1, concat=False)

    def forward(self, object_x, concept_x, obj_to_concept, concept_to_obj, edge_attr, rev_edge_attr):
        object_h0 = self.object_encoder(object_x)
        concept_h0 = self.concept_encoder(concept_x)
        concept_h = self.object_to_concept((object_h0, concept_h0), obj_to_concept, edge_attr)
        concept_h = F.dropout(F.relu(concept_h), p=self.dropout, training=self.training)
        object_msg = self.concept_to_object((concept_h, object_h0), concept_to_obj, rev_edge_attr)
        object_msg = F.dropout(F.relu(object_msg), p=self.dropout, training=self.training)
        return object_h0 + object_msg


class DualWeightedBipartiteTransformer(nn.Module):
    def __init__(
        self,
        pos_object_in_channels: int,
        neg_object_in_channels: int,
        pos_concept_channels: int,
        neg_concept_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.pos_branch = WeightedBipartiteBranch(pos_object_in_channels, pos_concept_channels, hidden_channels, heads, dropout)
        self.neg_branch = WeightedBipartiteBranch(neg_object_in_channels, neg_concept_channels, hidden_channels, heads, dropout)
        self.fusion_layer = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, batch: dict) -> torch.Tensor:
        pos_h = self.pos_branch(
            batch["x_pos"],
            batch["pos_concept_x"],
            batch["pos_edges"]["obj_to_concept"],
            batch["pos_edges"]["concept_to_obj"],
            batch["pos_edges"]["edge_attr"],
            batch["pos_edges"]["rev_edge_attr"],
        )
        neg_h = self.neg_branch(
            batch["x_neg"],
            batch["neg_concept_x"],
            batch["neg_edges"]["obj_to_concept"],
            batch["neg_edges"]["concept_to_obj"],
            batch["neg_edges"]["edge_attr"],
            batch["neg_edges"]["rev_edge_attr"],
        )
        return self.fusion_layer(torch.cat([pos_h, neg_h], dim=1))


def train_bipartite(config: TrainConfig, *, with_cpe: bool) -> TrainResult:
    set_seed(config.seed)
    batch = load_bipartite_batch(config, with_cpe=with_cpe)
    model = DualWeightedBipartiteTransformer(
        pos_object_in_channels=batch["x_pos"].shape[1],
        neg_object_in_channels=batch["x_neg"].shape[1],
        pos_concept_channels=batch["pos_concept_x"].shape[1],
        neg_concept_channels=batch["neg_concept_x"].shape[1],
        hidden_channels=config.hidden_channels,
        out_channels=batch["num_classes"],
        heads=config.heads,
        dropout=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    return run_training_loop(
        config,
        model,
        optimizer,
        criterion,
        forward_fn=lambda: model(batch),
        y=batch["y"],
        train_mask=batch["train_mask"],
        val_mask=batch["val_mask"],
        test_mask=batch["test_mask"],
    )


def accuracy(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    return (pred[mask] == y[mask]).sum().item() / mask.sum().item()


def run_training_loop(
    config: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    forward_fn,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> TrainResult:
    run_dir = make_run_dir(config)
    writer = SummaryWriter(str(run_dir))
    epoch_metrics_path = run_dir / "epoch_metrics.csv"
    best_val = (-1.0, 0)
    best_test = (-1.0, 0)
    final_train_acc = final_val_acc = final_test_acc = 0.0

    print(f"TensorBoard log dir: {run_dir}")
    print(f"Start training: {config.experiment}, epochs={config.epochs}")

    try:
        with epoch_metrics_path.open("w", newline="", encoding="utf-8") as f:
            metrics_writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "train_acc", "val_acc", "test_acc"])
            metrics_writer.writeheader()

            for epoch in range(1, config.epochs + 1):
                model.train()
                optimizer.zero_grad()
                out = forward_fn()
                loss = criterion(out[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()
                writer.add_scalar("Loss/train", loss.item(), epoch)

                model.eval()
                with torch.no_grad():
                    out = forward_fn()
                    pred = out.argmax(dim=1)
                    final_train_acc = accuracy(pred, y, train_mask)
                    final_val_acc = accuracy(pred, y, val_mask)
                    final_test_acc = accuracy(pred, y, test_mask)

                writer.add_scalar("Accuracy/train", final_train_acc, epoch)
                writer.add_scalar("Accuracy/validation", final_val_acc, epoch)
                writer.add_scalar("Accuracy/test", final_test_acc, epoch)
                metrics_writer.writerow(
                    {
                        "epoch": epoch,
                        "loss": loss.item(),
                        "train_acc": final_train_acc,
                        "val_acc": final_val_acc,
                        "test_acc": final_test_acc,
                    }
                )

                if final_val_acc > best_val[0]:
                    best_val = (final_val_acc, epoch)
                if final_test_acc > best_test[0]:
                    best_test = (final_test_acc, epoch)
                if epoch % config.log_every == 0 or epoch == 1 or epoch == config.epochs:
                    prefix = f"topK={config.topk}, " if config.topk is not None else ""
                    print(
                        f"{prefix}Epoch: {epoch:03d}, Loss: {loss.item():.4f}, "
                        f"Train Acc: {final_train_acc:.4f}, Val Acc: {final_val_acc:.4f}, Test Acc: {final_test_acc:.4f}"
                    )

        result_json_path = experiment_root(config) / "results" / f"{run_dir.name}.json"
        result = TrainResult(
            dataset=config.dataset,
            experiment=config.experiment,
            topk=config.topk,
            final_train_acc=final_train_acc,
            final_val_acc=final_val_acc,
            final_test_acc=final_test_acc,
            best_val_acc=best_val[0],
            best_val_epoch=best_val[1],
            best_test_acc=best_test[0],
            best_test_epoch=best_test[1],
            log_dir=str(run_dir),
            epoch_metrics_path=str(epoch_metrics_path),
            result_json_path=str(result_json_path),
        )
        serializable_config = asdict(config)
        serializable_config["input_dir"] = str(config.input_dir)
        serializable_config["output_dir"] = str(config.output_dir)
        with result_json_path.open("w", encoding="utf-8") as f:
            json.dump({"config": serializable_config, "result": asdict(result)}, f, indent=2, ensure_ascii=False)
        append_summary(config, result)
        writer.add_hparams(
            {k: v for k, v in serializable_config.items() if isinstance(v, (int, float, str, bool))},
            {
                "accuracy/final_train": final_train_acc,
                "accuracy/final_validation": final_val_acc,
                "accuracy/final_test": final_test_acc,
            },
        )
        print(f"Finished {config.experiment}: final_test={final_test_acc:.4f}, best_test={best_test[0]:.4f}@{best_test[1]}")
        return result
    finally:
        writer.close()


def append_summary(config: TrainConfig, result: TrainResult) -> None:
    summary_path = experiment_root(config) / "results" / "summary.csv"
    row = asdict(result)
    exists = summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_suite_summary(output_dir: Path, dataset: str, results: Iterable[TrainResult]) -> Path:
    root = output_dir / f"{dataset}_output" / "results"
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"suite_summary_{timestamp}.csv"
    rows = [asdict(result) for result in results]
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
