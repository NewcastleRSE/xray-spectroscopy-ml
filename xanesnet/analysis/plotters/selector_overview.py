# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   ...

"""Plotter that clusters structures by descriptor similarity and scores the clusters by error."""

import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, PowerNorm
from scipy.cluster.hierarchy import fcluster, linkage

from xanesnet.descriptors import DescriptorRegistry
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from ..reporters.base import selector_label
from ..result import AnalysisResults
from ..selectors import Selector
from .base import Plotter
from .registry import PlotterRegistry
from .utils import (
    _draw_structure,
    add_subtitle,
    compact_layout,
    method_colour,
    method_label_lines,
    spectrum_error_value,
    style_axis,
)

# Sample id, feature vector, error, prediction sample, target site index.
_SampleInfo = tuple[str, np.ndarray, float, PredictionSample, int | None]

# Cluster id, size, mean error, median error.
_ClusterStats = tuple[int, int, float, float]


@PlotterRegistry.register("structure_clusters")
class StructureClusterPlotter(Plotter):
    """Cluster structures by descriptor similarity and score the clusters by error.

    For every (prediction-reader, selector) pair the selected samples with an
    attached structure are embedded with a XANESNET descriptor (created via
    ``DescriptorRegistry``). Hierarchical clustering groups the structures
    into families whose count emerges from a linkage-distance threshold, and
    each family is scored by its per-sample error. This shows which structure
    families the model predicts well and which it fails on.

    Three figures are written per method: an error map (2D PCA projection
    coloured by error), a cluster scoreboard (mean error per cluster against
    the global mean), and detail pages for the best and worst clusters with
    representative structures and their mean spectra.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        descriptor_type: Registered descriptor name from ``xanesnet.descriptors``.
        cluster_distance: Linkage-distance cut for cluster formation. ``None``
            uses ``distance_fraction`` of the maximum linkage height.
        distance_fraction: Fraction of the maximum linkage height used as the
            cut when ``cluster_distance`` is ``None``. Lower values produce
            more clusters.
        min_cluster_size: Clusters with fewer members are folded into an
            ``others`` bucket.
        sort_key: Scalar key used as the per-sample error. When ``None``, the
            MSE between predicted and target spectra is used. Collector values
            take precedence over sample scalars.
        top_clusters: Number of best and worst clusters shown on the detail pages.
        **descriptor_kwargs: Keyword arguments forwarded to the configured descriptor.
    """

    def __init__(
        self,
        plotter_type: str,
        descriptor_type: str = "wacsf",
        cluster_distance: float | None = None,
        distance_fraction: float = 0.25,
        min_cluster_size: int = 3,
        sort_key: str | None = None,
        top_clusters: int = 5,
        **descriptor_kwargs: Any,
    ) -> None:
        """Initialize a structure cluster plotter.

        Raises:
            ValueError: If ``distance_fraction`` is not within ``(0, 1)``, or
                ``min_cluster_size`` or ``top_clusters`` is below one.
        """
        super().__init__(plotter_type)
        if not 0.0 < distance_fraction < 1.0:
            raise ValueError(f"distance_fraction must be in (0, 1), got {distance_fraction}")
        if min_cluster_size < 1:
            raise ValueError(f"min_cluster_size must be at least 1, got {min_cluster_size}")
        if top_clusters < 1:
            raise ValueError(f"top_clusters must be at least 1, got {top_clusters}")
        self.descriptor = DescriptorRegistry.create(
            descriptor_type, descriptor_type=descriptor_type, **descriptor_kwargs
        )
        self.cluster_distance = cluster_distance
        self.distance_fraction = distance_fraction
        self.min_cluster_size = min_cluster_size
        self.sort_key = sort_key
        self.top_clusters = top_clusters

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write per-method structure cluster figures.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``structure_clusters`` tree should be written.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "structure_clusters"
        root.mkdir(parents=True, exist_ok=True)
        err_key = self.sort_key if self.sort_key is not None else "mse"

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                sel_label_str = selector_label(results.selectors_config, sel_idx)
                sel_cfg = results.selectors_config[sel_idx]
                label_lines = method_label_lines(results.prediction_names[reader_idx], sel_label_str, sel_cfg)

                stream: JSONLStream | None = None
                if reader_idx < len(results.collector_results) and sel_idx < len(results.collector_results[reader_idx]):
                    stream = results.collector_results[reader_idx][sel_idx]

                samples, missing = self._collect_samples(selector, stream)
                if missing:
                    logging.info(f"      Skipped {missing} samples without structures.")
                if len(samples) < 2:
                    logging.info("      Need at least two structures for clustering, skipping.")
                    continue

                features = np.stack([sample[1] for sample in samples])
                scaled = self._standardize(features)
                labels, n_clusters = self._cluster(
                    scaled, self.cluster_distance, self.distance_fraction, self.min_cluster_size
                )
                errors = np.array([sample[2] for sample in samples])
                stats = self._cluster_stats(labels, errors, n_clusters)
                projection = self._pca_2d(scaled)

                colour = method_colour(reader_idx * len(reader_selectors) + sel_idx)
                combo_label = f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{sel_label_str}"
                combo_dir = root / combo_label
                combo_dir.mkdir(parents=True, exist_ok=True)
                subtitle = "\n".join(label_lines)

                self._error_map(projection, errors, labels, n_clusters, subtitle, err_key, combo_dir / "error_map.pdf")
                self._error_map_3d(
                    scaled, errors, labels, n_clusters, subtitle, err_key, combo_dir / "error_map_3d.pdf"
                )
                self._scoreboard(stats, float(errors.mean()), err_key, subtitle, combo_dir / "cluster_scoreboard.pdf")
                self._cluster_pages(
                    samples,
                    labels,
                    scaled,
                    stats,
                    colour,
                    label_lines,
                    err_key,
                    combo_dir / "cluster_pages.pdf",
                )

    def _collect_samples(self, selector: Selector, stream: JSONLStream | None) -> tuple[list[_SampleInfo], int]:
        """Embed every selected sample with a structure using the configured descriptor.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.

        Returns:
            ``(samples, missing)`` where ``samples`` holds the embedded
            ``(sample id, features, error, sample, site index)`` records and
            ``missing`` counts the selected samples without a structure.
        """
        samples: list[_SampleInfo] = []
        missing = 0
        pairs = zip(selector, stream) if stream is not None else ((sample, {}) for sample in selector)
        for sel_sample, col_sample in pairs:
            structure = sel_sample.get("structure")
            if structure is None:
                missing += 1
                continue
            site_index = sel_sample.get("target_site_index")
            vector = np.asarray(self.descriptor.transform_pmg(structure, site_index=site_index), dtype=float)
            if vector.ndim == 2:
                vector = vector.mean(axis=0)
            samples.append(
                (
                    str(sel_sample["sample_id"]),
                    vector.ravel(),
                    spectrum_error_value(sel_sample, col_sample, self.sort_key),
                    sel_sample,
                    site_index,
                )
            )
        return samples, missing

    @staticmethod
    def _standardize(features: np.ndarray) -> np.ndarray:
        """Center feature columns and scale them to unit variance.

        Args:
            features: Feature matrix with shape ``(M, D)``.

        Returns:
            Standardized feature matrix with shape ``(M, D)``.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = (features - features.mean(axis=0)) / features.std(axis=0)
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _cluster(
        features: np.ndarray,
        cluster_distance: float | None,
        distance_fraction: float,
        min_cluster_size: int,
    ) -> tuple[np.ndarray, int]:
        """Cluster samples with agglomerative hierarchical clustering.

        The cluster count emerges from a linkage-distance threshold. Clusters
        smaller than ``min_cluster_size`` are folded into the ``others`` label
        ``-1``; real clusters are renumbered zero-based in first-seen order.

        Args:
            features: Standardized feature matrix with shape ``(M, D)``.
            cluster_distance: Linkage-distance cut, or ``None`` to use
                ``distance_fraction`` of the maximum linkage height.
            distance_fraction: Fraction of the maximum linkage height used as
                the cut; lower values produce more clusters.
            min_cluster_size: Minimum cluster size before folding into ``others``.

        Returns:
            ``(labels, n_clusters)`` with ``labels`` of shape ``(M,)``.
        """
        if len(features) == 1:
            return np.zeros(1, dtype=int), 0

        z = linkage(features, method="ward")
        threshold = cluster_distance if cluster_distance is not None else distance_fraction * float(z[-1, 2])
        raw = fcluster(z, t=threshold, criterion="distance")

        counts = np.bincount(raw)
        small = {i + 1 for i, count in enumerate(counts) if count > 0 and count < min_cluster_size}
        mapping: dict[int, int] = {}
        for raw_id in raw:
            key = int(raw_id) - 1
            if int(raw_id) not in small and key not in mapping:
                mapping[key] = len(mapping)
        labels = np.array([-1 if int(raw_id) in small else mapping[int(raw_id) - 1] for raw_id in raw], dtype=int)
        return labels, len(mapping)

    @staticmethod
    def _cluster_stats(labels: np.ndarray, errors: np.ndarray, n_clusters: int) -> list[_ClusterStats]:
        """Compute size, mean error, and median error per cluster.

        Args:
            labels: Cluster label per sample; ``-1`` marks the ``others`` bucket.
            errors: Per-sample error values with shape ``(M,)``.
            n_clusters: Number of real clusters.

        Returns:
            ``(cluster id, size, mean error, median error)`` entries, with
            ``-1`` for the ``others`` bucket when it is non-empty.
        """
        stats: list[_ClusterStats] = []
        for cid in range(n_clusters):
            mask = labels == cid
            if mask.sum() == 0:
                continue
            stats.append((cid, int(mask.sum()), float(errors[mask].mean()), float(np.median(errors[mask]))))
        others = labels == -1
        if others.any():
            stats.append((-1, int(others.sum()), float(errors[others].mean()), float(np.median(errors[others]))))
        return stats

    @staticmethod
    def _pca_2d(features: np.ndarray) -> np.ndarray:
        """Project standardized features onto their first two principal components.

        Args:
            features: Standardized feature matrix with shape ``(M, D)``.

        Returns:
            Projection with shape ``(M, 2)``.
        """
        centered = features - features.mean(axis=0)
        if centered.shape[1] < 2:
            return np.column_stack([centered[:, 0], np.zeros(len(features))])
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt[:2].T

    @staticmethod
    def _pca_3d(features: np.ndarray) -> np.ndarray:
        """Project standardized features onto their first three principal components.

        Args:
            features: Standardized feature matrix with shape ``(M, D)``.

        Returns:
            Projection with shape ``(M, 3)``; missing components are zero-filled.
        """
        centered = features - features.mean(axis=0)
        if centered.shape[1] < 2:
            return np.column_stack([centered[:, 0], np.zeros((len(features), 2))])
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[: min(3, vt.shape[0])].T
        if proj.shape[1] < 3:
            proj = np.column_stack([proj, np.zeros((len(features), 3 - proj.shape[1]))])
        return proj

    @staticmethod
    def _error_norm(errors: np.ndarray) -> Normalize:
        """Build a colour normalization that spreads the error scale.

        Per-sample errors are usually right-skewed, so a plain linear
        normalization lets a few large values compress everything else into a
        single colour. Clipping to the 2nd-98th percentiles and applying a
        power scale (gamma 0.4) spreads the low end where most samples live.

        Args:
            errors: Per-sample error values with shape ``(M,)``.

        Returns:
            Matplotlib normalization for colouring the error map.
        """
        errors = np.asarray(errors, dtype=float)
        lo = float(np.percentile(errors, 2))
        hi = float(np.percentile(errors, 98))
        if hi - lo < 1e-12 or lo < 0:
            return Normalize(vmin=float(errors.min()), vmax=float(errors.max()) + 1e-12)
        return PowerNorm(gamma=0.4, vmin=lo, vmax=hi, clip=True)

    def _error_map(
        self,
        projection: np.ndarray,
        errors: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        subtitle: str,
        err_key: str,
        out: Path,
    ) -> None:
        """Write the 2D error map with cluster centroid markers.

        Args:
            projection: PCA projection with shape ``(M, 2)``.
            errors: Per-sample error values with shape ``(M,)``.
            labels: Cluster label per sample; ``-1`` marks the ``others`` bucket.
            n_clusters: Number of real clusters.
            subtitle: Plot subtitle text describing prediction and selector context.
            err_key: Scalar key used for the colour scale.
            out: Destination PDF path.
        """
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        if (labels == -1).any():
            ax.scatter(
                projection[labels == -1, 0],
                projection[labels == -1, 1],
                color="gray",
                s=12,
                alpha=0.5,
                zorder=1,
            )
        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            c=errors,
            cmap="viridis",
            norm=self._error_norm(errors),
            s=18,
            alpha=0.85,
            zorder=2,
        )
        for cid in range(n_clusters):
            mask = labels == cid
            if mask.sum() == 0:
                continue
            centre = projection[mask].mean(axis=0)
            ax.annotate(
                str(cid + 1),
                (centre[0], centre[1]),
                fontsize=9,
                fontweight="bold",
                ha="center",
                va="center",
                color="black",
                bbox=dict(boxstyle="circle,pad=0.25", facecolor="white", alpha=0.85, edgecolor="black"),
            )
        fig.colorbar(scatter, ax=ax, label=err_key)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        style_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    def _error_map_3d(
        self,
        features: np.ndarray,
        errors: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
        subtitle: str,
        err_key: str,
        out: Path,
    ) -> None:
        """Write the 3D error map with cluster centroid markers.

        Args:
            features: Standardized feature matrix with shape ``(M, D)``.
            errors: Per-sample error values with shape ``(M,)``.
            labels: Cluster label per sample; ``-1`` marks the ``others`` bucket.
            n_clusters: Number of real clusters.
            subtitle: Plot subtitle text describing prediction and selector context.
            err_key: Scalar key used for the colour scale.
            out: Destination PDF path.
        """
        from mpl_toolkits.mplot3d import (  # noqa: F401  (registers the 3D projection)
            Axes3D,
        )

        projection = self._pca_3d(features)
        norm = self._error_norm(errors)

        fig = plt.figure(figsize=(6.5, 5.4))
        ax = cast(Any, fig.add_subplot(111, projection="3d"))
        if (labels == -1).any():
            ax.scatter(
                projection[labels == -1, 0],
                projection[labels == -1, 1],
                projection[labels == -1, 2],
                color="gray",
                s=10,
                alpha=0.5,
                depthshade=False,
            )
        scatter = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            projection[:, 2],
            c=errors,
            cmap="viridis",
            norm=norm,
            s=14,
            alpha=0.85,
            depthshade=False,
        )
        for cid in range(n_clusters):
            mask = labels == cid
            if mask.sum() == 0:
                continue
            centre = projection[mask].mean(axis=0)
            ax.text(
                centre[0],
                centre[1],
                centre[2],
                str(cid + 1),
                fontsize=9,
                fontweight="bold",
                color="black",
                zorder=10,
            )
        fig.colorbar(scatter, ax=ax, label=err_key, shrink=0.7, pad=0.08)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        style_axis(ax)
        add_subtitle(fig, subtitle)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _scoreboard(
        stats: list[_ClusterStats],
        global_mean: float,
        err_key: str,
        subtitle: str,
        out: Path,
    ) -> None:
        """Write the cluster scoreboard sorted by mean error.

        Bars below the global mean error are green, bars above it are red, and
        the ``others`` bucket is grey at the bottom.

        Args:
            stats: Per-cluster ``(id, size, mean, median)`` entries.
            global_mean: Mean error over all selected samples.
            err_key: Scalar key shown on the x axis.
            subtitle: Plot subtitle text describing prediction and selector context.
            out: Destination PDF path.
        """
        rows = sorted([s for s in stats if s[0] != -1], key=lambda s: s[2])
        if not rows:
            return
        labels = [f"cluster {cid + 1} (n={size})" for cid, size, _, _ in rows]
        means = [mean for _, _, mean, _ in rows]
        colours = ["#3f9e6e" if mean <= global_mean else "#e85651" for mean in means]
        others = next((s for s in stats if s[0] == -1), None)
        if others is not None:
            labels.append(f"others (n={others[1]})")
            means.append(others[2])
            colours.append("gray")

        fig, ax = plt.subplots(figsize=(6.5, max(2.4, 0.35 * len(labels) + 0.9)))
        ys = np.arange(len(means))
        ax.barh(ys, means, color=colours, alpha=0.8)
        ax.axvline(global_mean, color="black", linewidth=1.0, linestyle="--", label="global mean")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(err_key)
        ax.legend(fontsize=8, framealpha=0.9)
        style_axis(ax)
        add_subtitle(fig, subtitle)
        compact_layout(fig)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    def _cluster_pages(
        self,
        samples: list[_SampleInfo],
        labels: np.ndarray,
        scaled: np.ndarray,
        stats: list[_ClusterStats],
        colour: str,
        label_lines: list[str],
        err_key: str,
        out: Path,
    ) -> None:
        """Write detail pages for the best and worst clusters.

        Args:
            samples: Embedded ``(sample id, features, error, sample, site index)`` records.
            labels: Cluster label per sample; ``-1`` marks the ``others`` bucket.
            scaled: Standardized feature matrix with shape ``(M, D)``.
            stats: Per-cluster ``(id, size, mean, median)`` entries.
            colour: Method colour used for the mean prediction curve.
            label_lines: Method label lines used for the page subtitle.
            err_key: Scalar key shown in the page annotations.
            out: Destination PDF path.
        """
        real = sorted([s for s in stats if s[0] != -1], key=lambda s: s[2])
        if not real:
            return
        best = real[: self.top_clusters]
        worst = list(reversed(real[-self.top_clusters :]))

        with PdfPages(out) as pdf:
            for rank, (cid, size, mean_err, median_err) in enumerate(best, start=1):
                fig = self._cluster_page(
                    samples,
                    labels,
                    scaled,
                    cid,
                    size,
                    mean_err,
                    median_err,
                    f"best cluster #{rank} of {len(real)}",
                    colour,
                    label_lines,
                    err_key,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            for rank, (cid, size, mean_err, median_err) in enumerate(worst, start=1):
                fig = self._cluster_page(
                    samples,
                    labels,
                    scaled,
                    cid,
                    size,
                    mean_err,
                    median_err,
                    f"worst cluster #{rank} of {len(real)}",
                    colour,
                    label_lines,
                    err_key,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    def _cluster_page(
        self,
        samples: list[_SampleInfo],
        labels: np.ndarray,
        scaled: np.ndarray,
        cid: int,
        size: int,
        mean_err: float,
        median_err: float,
        direction_text: str,
        colour: str,
        label_lines: list[str],
        err_key: str,
    ) -> Any:
        """Build one cluster detail page.

        The left panel shows the cluster mean prediction and target spectra and
        the right panel a grid of representative structures (the members
        closest to the cluster centroid).

        Args:
            samples: Embedded ``(sample id, features, error, sample, site index)`` records.
            labels: Cluster label per sample; ``-1`` marks the ``others`` bucket.
            scaled: Standardized feature matrix with shape ``(M, D)``.
            cid: Cluster id to render.
            size: Number of cluster members.
            mean_err: Mean error of the cluster.
            median_err: Median error of the cluster.
            direction_text: Best/worst ranking text for the page subtitle.
            colour: Method colour used for the mean prediction curve.
            label_lines: Method label lines used for the page subtitle.
            err_key: Scalar key shown in the page annotations.

        Returns:
            Matplotlib figure for the cluster detail page.
        """
        members = [samples[i] for i, label in enumerate(labels) if label == cid]
        centroid = np.mean([scaled[i] for i, label in enumerate(labels) if label == cid], axis=0)
        member_indices = [i for i, label in enumerate(labels) if label == cid]
        order = sorted(member_indices, key=lambda i: float(np.linalg.norm(scaled[i] - centroid)))
        reps = [samples[i] for i in order[:9]]

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.45])

        ax_spec = fig.add_subplot(gs[0, 0])
        preds = np.stack([np.asarray(sample[3]["prediction"]).ravel() for sample in members])
        targets = np.stack([np.asarray(sample[3]["target"]).ravel() for sample in members])
        target_mean = targets.mean(axis=0)
        pred_mean = preds.mean(axis=0)
        pred_std = preds.std(axis=0)
        x = np.arange(len(target_mean))
        ax_spec.plot(x, target_mean, color="black", linewidth=1.6, label="Target mean")
        ax_spec.fill_between(
            x,
            pred_mean - pred_std,
            pred_mean + pred_std,
            color=colour,
            alpha=0.25,
            linewidth=0,
            label="Prediction +/- 1 std",
        )
        ax_spec.plot(x, pred_mean, color=colour, linewidth=2.0, label="Prediction mean")
        ax_spec.set_xlabel("Energy")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title(
            f"Cluster {cid + 1}: n={size}, mean {err_key}={mean_err:.4g}, median={median_err:.4g}",
            loc="left",
        )
        ax_spec.legend(fontsize=8, framealpha=0.9)
        style_axis(ax_spec)

        gs_right = gs[0, 1].subgridspec(3, 3, wspace=0.04, hspace=0.14)
        for k, (sample_id, _, error, sample, site_index) in enumerate(reps):
            ax = fig.add_subplot(gs_right[k // 3, k % 3])
            structure = sample.get("structure")
            if structure is not None:
                _draw_structure(ax, structure, sample_id, site_index)
            ax.set_title(f"{sample_id}  {err_key}={error:.3g}", fontsize=5.5)
        for k in range(len(reps), 9):
            fig.add_subplot(gs_right[k // 3, k % 3]).axis("off")

        add_subtitle(fig, f"{direction_text}  |  " + "  |  ".join(label_lines))
        fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.08, wspace=0.06)
        return fig
