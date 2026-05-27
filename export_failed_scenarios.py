import argparse
import logging
import os
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from bokeh.io.export import get_screenshot_as_png
from bokeh.layouts import column
from selenium import webdriver
from tqdm import tqdm

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory, get_maps_db
from nuplan.planning.nuboard.base.data_class import SimulationScenarioKey
from nuplan.planning.nuboard.base.experiment_file_data import ExperimentFileData
from nuplan.planning.nuboard.base.simulation_tile import SimulationTile
from nuplan.planning.nuboard.style import simulation_tile_style
from nuplan.planning.nuboard.utils.utils import read_nuboard_file_paths


class ImmediateDocument:
    """A tiny bokeh-like document that executes callbacks immediately."""

    def add_next_tick_callback(self, callback):
        callback()
        return None

    def add_periodic_callback(self, callback, period_milliseconds):
        return None

    def remove_periodic_callback(self, callback_handle):
        return None


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("render_zero_score_scenarios")


def collect_nuboard_files(result_dir: Path) -> List[Path]:
    return sorted(result_dir.rglob("*.nuboard"))


def _is_scenario_row(row: pd.Series) -> bool:
    # In nuBoard aggregator parquet, scenario-level rows have num_scenarios = NaN
    value = row.get("num_scenarios", np.nan)
    return pd.isna(value)


def collect_zero_score_keys(metric_aggregator_dataframes: dict, logger: logging.Logger) -> Set[Tuple[str, str, str, str]]:
    """
    Return a set of (planner_name, scenario_type, log_name, scenario_name) where score == 0.
    """
    keys: Set[Tuple[str, str, str, str]] = set()

    for aggregator_name, df in metric_aggregator_dataframes.items():
        required_cols = {"planner_name", "scenario_type", "log_name", "scenario", "score"}
        if not required_cols.issubset(set(df.columns)):
            logger.warning("Skip aggregator '%s': missing required columns %s", aggregator_name, required_cols - set(df.columns))
            continue

        for _, row in df.iterrows():
            if not _is_scenario_row(row):
                continue

            score = row["score"]
            if pd.isna(score):
                continue

            if float(score) == 0.0:
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
    zero_score_keys: Set[Tuple[str, str, str, str]],
) -> List[SimulationScenarioKey]:
    filtered: List[SimulationScenarioKey] = []
    for key in simulation_scenario_keys:
        if (key.planner_name, key.scenario_type, key.log_name, key.scenario_name) in zero_score_keys:
            filtered.append(key)
    return filtered


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
    video_name = f"{scenario_key.scenario_type}_{scenario_key.planner_name}_{scenario_key.log_name}_{scenario_key.scenario_name}.avi"
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
        tile._render_plots(main_figure=figure, frame_index=frame_idx)  # keep ego-centered each frame
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


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选score=0场景并按nuBoard风格渲染视频")
    parser.add_argument("--result_dir", type=str, required=True, help="实验结果目录（包含.nuboard）")
    parser.add_argument(
        "--map_root",
        type=str,
        default="/home/skt/Code/nuplan-devkit/nuplan/dataset/maps",
        help="地图根目录（与run_nuboard.py一致）",
    )
    parser.add_argument(
        "--map_version",
        type=str,
        default="nuplan-maps-v1.0",
        help="地图版本",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出视频目录，默认 <result_dir>/video_zero_score",
    )
    args = parser.parse_args()

    logger = setup_logging()

    # Keep the same env style as run_nuboard.py
    os.environ.setdefault("NUPLAN_DEVKIT_ROOT", "/home/skt/Code/nuplan-devkit")
    os.environ.setdefault("NUPLAN_DATA_ROOT", "/home/skt/Code/nuplan-devkit/nuplan/dataset")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", args.map_root)
    os.environ.setdefault("NUPLAN_EXP_ROOT", "/home/skt/Code/nuplan-devkit/nuplan/exp")
    os.environ.setdefault("NUPLAN_SIMULATION_ALLOW_ANY_BUILDER", "1")

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"result_dir not found: {result_dir}")

    nuboard_files = collect_nuboard_files(result_dir)
    if not nuboard_files:
        raise FileNotFoundError(f"No .nuboard found under: {result_dir}")

    logger.info("Found %d .nuboard file(s)", len(nuboard_files))

    # Usually one experiment = one .nuboard, but support multiple.
    output_dir = Path(args.output_dir) if args.output_dir else (result_dir / "video_zero_score")
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_db = get_maps_db(args.map_root, args.map_version)
    map_factory = NuPlanMapFactory(maps_db)

    rendered = 0
    driver = build_chrome_driver()
    try:
        for nuboard_path in nuboard_files:
            logger.info("Loading experiment from %s", nuboard_path)
            # Match run_nuboard: set current_path to the .nuboard parent so metrics/simulation
            # load from the experiment folder even when stored paths in the file are stale.
            nuboard_file = read_nuboard_file_paths([nuboard_path])[0]
            experiment_data = ExperimentFileData(file_paths=[nuboard_file])
            file_index = len(experiment_data.file_paths) - 1
            zero_score_keys = collect_zero_score_keys(
                experiment_data.metric_aggregator_dataframes[file_index], logger
            )
            selected_keys = filter_simulation_keys(experiment_data.simulation_scenario_keys, zero_score_keys)

            logger.info(
                "Experiment %s: zero-score scenario keys = %d, renderable simulation keys = %d",
                nuboard_path.name,
                len(zero_score_keys),
                len(selected_keys),
            )

            existing_videos = sum(
                (
                    output_dir
                    / f"{key.scenario_type}_{key.planner_name}_{key.log_name}_{key.scenario_name}.avi"
                ).exists()
                for key in selected_keys
            )
            logger.info(
                "Experiment %s: existing videos = %d, pending renders = %d",
                nuboard_path.name,
                existing_videos,
                len(selected_keys) - existing_videos,
            )

            for key in tqdm(selected_keys, desc=f"Rendering {nuboard_path.stem}"):
                render_scenario_video(
                    scenario_key=key,
                    experiment_file_data=experiment_data,
                    map_factory=map_factory,
                    output_dir=output_dir,
                    driver=driver,
                    logger=logger,
                )
                rendered += 1

    finally:
        driver.quit()

    logger.info("Done. Rendered %d video(s). Output dir: %s", rendered, output_dir)


if __name__ == "__main__":
    main()
