"""
Shared helpers for export_failed_scenarios.py and export_success_scenarios.py.

Supports --result_dir as:
  - a single nuPlan experiment folder (contains aggregator_metric + simulation_log), or
  - a parent directory that contains one or more such experiment folders (recursive).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from bokeh.io.export import get_screenshot_as_png
from bokeh.layouts import column
from selenium import webdriver
from tqdm import tqdm

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
from nuplan.planning.nuboard.base.data_class import NuBoardFile, SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.base.simulation_tile import SimulationTile
from nuplan.planning.nuboard.style import simulation_tile_style
from nuplan.planning.nuboard.utils.utils import read_nuboard_file_paths

ScenarioKey = Tuple[str, str, str, str]  # planner_name, scenario_type, log_name, scenario_name

EXPERIMENT_MARKERS = ("aggregator_metric", "simulation_log")


class ImmediateDocument:
    """A tiny bokeh-like document that executes callbacks immediately."""

    def add_next_tick_callback(self, callback):
        callback()
        return None

    def add_periodic_callback(self, callback, period_milliseconds):
        return None

    def remove_periodic_callback(self, callback_handle):
        return None


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    return logging.getLogger(name)


def setup_nuplan_env(map_root: str) -> None:
    os.environ.setdefault("NUPLAN_DEVKIT_ROOT", "/home/skt/Code/nuplan-devkit")
    os.environ.setdefault("NUPLAN_DATA_ROOT", "/home/skt/Code/nuplan-devkit/nuplan/dataset")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", map_root)
    os.environ.setdefault("NUPLAN_EXP_ROOT", "/home/skt/Code/nuplan-devkit/nuplan/exp")
    os.environ.setdefault("NUPLAN_SIMULATION_ALLOW_ANY_BUILDER", "1")


def is_experiment_root(path: Path) -> bool:
    return path.is_dir() and all((path / marker).is_dir() for marker in EXPERIMENT_MARKERS)


def resolve_experiment_roots(result_dir: Path) -> List[Path]:
    """
    Return experiment root directories under result_dir.

    If result_dir itself is an experiment root, return only that directory.
    Otherwise collect every descendant directory that looks like an experiment root.
    """
    result_dir = result_dir.resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir not found: {result_dir}")

    if is_experiment_root(result_dir):
        return [result_dir]

    roots: List[Path] = []
    for path in sorted(result_dir.rglob("*")):
        if path.is_dir() and is_experiment_root(path):
            roots.append(path.resolve())

    if not roots:
        raise FileNotFoundError(
            f"No nuPlan experiment directory found under {result_dir}. "
            f"Expected subfolders {EXPERIMENT_MARKERS}."
        )
    return roots


def find_nuboard_file(experiment_root: Path) -> Optional[Path]:
    files = sorted(experiment_root.glob("*.nuboard"))
    return files[-1] if files else None


def build_nuboard_file(experiment_root: Path, nuboard_path: Optional[Path] = None) -> NuBoardFile:
    """
    Load .nuboard if present; always set current_path to experiment_root so metrics/simulation
    are read from the actual result folder (stored paths inside .nuboard may be stale).
    """
    experiment_root = experiment_root.resolve()
    if nuboard_path is not None and nuboard_path.exists():
        nuboard_file = read_nuboard_file_paths([nuboard_path])[0]
    else:
        nuboard_file = NuBoardFile(
            simulation_main_path=str(experiment_root),
            metric_main_path=str(experiment_root),
            metric_folder="metrics",
            aggregator_metric_folder="aggregator_metric",
            simulation_folder="simulation_log",
        )
    nuboard_file.current_path = experiment_root
    return nuboard_file


def load_experiment_data(experiment_root: Path, logger: logging.Logger) -> Tuple[NuBoardFile, ExperimentFileData, int]:
    experiment_root = experiment_root.resolve()
    nuboard_path = find_nuboard_file(experiment_root)
    if nuboard_path is not None:
        logger.info("Experiment root: %s (nuboard: %s)", experiment_root, nuboard_path.name)
    else:
        logger.warning(
            "No .nuboard under %s; loading metrics/simulation directly from experiment folders.",
            experiment_root,
        )
    nuboard_file = build_nuboard_file(experiment_root, nuboard_path)
    experiment_data = ExperimentFileData(file_paths=[nuboard_file])
    file_index = len(experiment_data.file_paths) - 1

    if not experiment_data.metric_aggregator_dataframes[file_index]:
        raise FileNotFoundError(
            f"No aggregator metrics under {experiment_root / nuboard_file.aggregator_metric_folder}"
        )
    if not experiment_data.simulation_scenario_keys:
        raise FileNotFoundError(
            f"No simulation logs under {experiment_root / nuboard_file.simulation_folder}"
        )
    return nuboard_file, experiment_data, file_index


def _is_scenario_row(row: pd.Series) -> bool:
    value = row.get("num_scenarios", np.nan)
    return pd.isna(value)


def collect_scenario_keys_by_score(
    metric_aggregator_dataframes: dict,
    logger: logging.Logger,
    score_predicate: Callable[[float], bool],
) -> Set[ScenarioKey]:
    keys: Set[ScenarioKey] = set()

    for aggregator_name, df in metric_aggregator_dataframes.items():
        required_cols = {"planner_name", "scenario_type", "log_name", "scenario", "score"}
        if not required_cols.issubset(set(df.columns)):
            logger.warning(
                "Skip aggregator '%s': missing required columns %s",
                aggregator_name,
                required_cols - set(df.columns),
            )
            continue

        for _, row in df.iterrows():
            if not _is_scenario_row(row):
                continue
            score = row["score"]
            if pd.isna(score):
                continue
            if score_predicate(float(score)):
                keys.add(
                    (
                        str(row["planner_name"]),
                        str(row["scenario_type"]),
                        str(row["log_name"]),
                        str(row["scenario"]),
                    )
                )
    return keys


def filter_simulation_keys(
    simulation_scenario_keys: Sequence[SimulationScenarioKey],
    target_keys: Set[ScenarioKey],
) -> List[SimulationScenarioKey]:
    return [
        key
        for key in simulation_scenario_keys
        if (key.planner_name, key.scenario_type, key.log_name, key.scenario_name) in target_keys
    ]


def build_chrome_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1920, 1080)
    return driver


def render_scenario_video(
    scenario_key: SimulationScenarioKey,
    experiment_file_data: ExperimentFileData,
    map_factory: NuPlanMapFactory,
    output_dir: Path,
    driver: webdriver.Chrome,
    logger: logging.Logger,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = (
        f"{scenario_key.scenario_type}_{scenario_key.planner_name}_"
        f"{scenario_key.log_name}_{scenario_key.scenario_name}.avi"
    )
    video_path = output_dir / video_name
    if video_path.exists():
        logger.info("Skip existing video: %s", video_path)
        return video_path

    doc = ImmediateDocument()
    tile = SimulationTile(
        doc=doc,  # type: ignore[arg-type]
        experiment_file_data=experiment_file_data,
        vehicle_parameters=get_pacifica_parameters(),
        map_factory=map_factory,
        async_rendering=False,
        frame_rate_cap_hz=60,
    )

    tile.render_simulation_tiles(
        selected_scenario_keys=[scenario_key],
        figure_sizes=simulation_tile_style["render_figure_sizes"],
        hidden_glyph_names=None,
    )
    figure = tile.figures[0]

    total_frames = len(figure.simulation_history.data)
    if total_frames == 0:
        raise RuntimeError(f"No simulation frames found for scenario: {scenario_key.scenario_name}")

    frames: List[np.ndarray] = []
    for frame_idx in tqdm(
        range(total_frames),
        desc=f"Frames {scenario_key.scenario_name}",
        leave=False,
    ):
        tile._render_plots(main_figure=figure, frame_index=frame_idx)
        image = get_screenshot_as_png(column(figure.figure), driver=driver)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        frames.append(frame)

    database_interval = figure.scenario.database_interval
    fps = (1.0 / database_interval) if database_interval else 20.0

    h, w = frames[0].shape[:2]
    video = cv2.VideoWriter(
        filename=str(video_path),
        fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        fps=fps,
        frameSize=(w, h),
    )
    for frame in frames:
        video.write(frame)
    video.release()

    logger.info("Saved: %s", video_path)
    return video_path


def resolve_output_dir(
    experiment_root: Path,
    result_dir: Path,
    output_dir: Optional[Path],
    default_subdir: str,
    multiple_experiments: bool,
) -> Path:
    if output_dir is not None:
        if multiple_experiments:
            return output_dir / experiment_root.name
        return output_dir
    return experiment_root / default_subdir


def run_export(
    *,
    result_dir: Path,
    output_dir: Optional[Path],
    default_output_subdir: str,
    score_predicate: Callable[[float], bool],
    score_label: str,
    logger_name: str,
    map_root: str,
    map_version: str,
) -> None:
    logger = setup_logging(logger_name)
    setup_nuplan_env(map_root)

    experiment_roots = resolve_experiment_roots(result_dir)
    logger.info("Found %d experiment(s) under %s", len(experiment_roots), result_dir.resolve())

    maps_db = get_maps_db(map_root, map_version)
    map_factory = NuPlanMapFactory(maps_db)
    multiple_experiments = len(experiment_roots) > 1
    user_output_dir = Path(output_dir) if output_dir else None

    rendered = 0
    driver = build_chrome_driver()
    try:
        for experiment_root in experiment_roots:
            exp_output_dir = resolve_output_dir(
                experiment_root,
                result_dir,
                user_output_dir,
                default_output_subdir,
                multiple_experiments,
            )
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            _, experiment_data, file_index = load_experiment_data(experiment_root, logger)
            target_keys = collect_scenario_keys_by_score(
                experiment_data.metric_aggregator_dataframes[file_index],
                logger,
                score_predicate,
            )
            selected_keys = filter_simulation_keys(experiment_data.simulation_scenario_keys, target_keys)

            logger.info(
                "Experiment %s: %s scenario keys = %d, renderable simulation keys = %d, output = %s",
                experiment_root.name,
                score_label,
                len(target_keys),
                len(selected_keys),
                exp_output_dir,
            )

            if not selected_keys:
                logger.warning(
                    "No renderable scenarios for experiment %s (check score filter vs simulation_log).",
                    experiment_root.name,
                )
                continue

            existing_videos = sum(
                (
                    exp_output_dir
                    / f"{key.scenario_type}_{key.planner_name}_{key.log_name}_{key.scenario_name}.avi"
                ).exists()
                for key in selected_keys
            )
            logger.info(
                "Experiment %s: existing videos = %d, pending renders = %d",
                experiment_root.name,
                existing_videos,
                len(selected_keys) - existing_videos,
            )

            for key in tqdm(selected_keys, desc=f"Rendering {experiment_root.name}"):
                render_scenario_video(
                    scenario_key=key,
                    experiment_file_data=experiment_data,
                    map_factory=map_factory,
                    output_dir=exp_output_dir,
                    driver=driver,
                    logger=logger,
                )
                rendered += 1

    finally:
        driver.quit()

    logger.info("Done. Rendered %d video(s) from %d experiment(s).", rendered, len(experiment_roots))
