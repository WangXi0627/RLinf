# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import glob
import importlib
import json
import os
import re
import sys
from typing import Optional, Union

import gym
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf

from rlinf.envs.sim.libero.utils import (
    build_interleaved_eval_reset_state_ids,
    distribute_reset_state_ids_round_robin,
    get_benchmark_overridden,
    get_libero_image,
    get_libero_type,
    get_libero_wrist_image,
    quat2axisangle,
    record_completed_episode_task_stats,
)
from rlinf.envs.sim.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor
from rlinf.utils.logging import get_logger


def _repoint_libero_config(libero_module) -> None:
    """Point LIBERO's cached config at the package that is actually installed.

    LIBERO writes absolute paths into a config file under ``$HOME`` the first
    time it is imported and afterwards only reads it back, so a config left by
    another venv on the same machine silently redirects asset and init-state
    lookups to directories this venv does not have.
    """
    installed_root = os.path.dirname(os.path.abspath(libero_module.__file__))
    try:
        configured_root = libero_module.get_libero_path("benchmark_root")
    except Exception:
        configured_root = None
    if configured_root != installed_root:
        libero_module.set_libero_default_path(installed_root)


logger = get_logger()


def _read_bddl_language_and_goal(bddl_path: str):
    """Parse (:language ...) and a compact (:goal ...) summary from a BDDL file."""
    try:
        with open(bddl_path, "r", encoding="utf-8") as f:
            bddl_text = f.read()
    except OSError:
        return None, None
    lang_m = re.search(r"\(:language\s+([^)]+)\)", bddl_text)
    language = lang_m.group(1).strip() if lang_m else None
    goal_m = re.search(r"\(:goal\s*\n?\s*\(And\s*\(([^)]+)\)\)", bddl_text)
    if goal_m is None:
        goal_m = re.search(r"\(:goal[\s\S]*?\(And\s*\(([^)]+)\)\)", bddl_text)
    goal = goal_m.group(1).strip() if goal_m else None
    return language, goal


def _resolve_plus_bddl_paths(
    bddl_root: str, problem_folder: str, task_name: str
) -> tuple[str, str, bool]:
    """Resolve a LIBERO-Plus logical task to runtime and readable BDDL paths.

    Some LIBERO-Plus variants intentionally use a virtual file name containing
    ``_view_`` and ``_initstate_`` (and optionally ``_noise_``). ControlEnv
    parses that suffix at runtime, then opens the physical base BDDL whose name
    is the prefix before ``_view_``.
    """
    name = task_name[:-5] if task_name.endswith(".bddl") else task_name
    runtime_path = os.path.abspath(
        os.path.join(bddl_root, problem_folder, f"{name}.bddl")
    )
    if os.path.isfile(runtime_path):
        return runtime_path, runtime_path, False

    if "_view_" in name and "_initstate_" in name:
        source_name = name.split("_view_", 1)[0]
        source_path = os.path.abspath(
            os.path.join(bddl_root, problem_folder, f"{source_name}.bddl")
        )
        if os.path.isfile(source_path):
            return runtime_path, source_path, True

    return runtime_path, runtime_path, False


libero_type = get_libero_type()

if libero_type in ["pro", "plus"]:
    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
    LIBERO_PKG_NAME = f"libero{libero_type}"
    LIBERO_MAIN_MODULE_PATH = f"{LIBERO_PKG_NAME}.{LIBERO_PKG_NAME}"
    try:
        real_libero_pkg = importlib.import_module(LIBERO_PKG_NAME)
        real_libero_core = importlib.import_module(LIBERO_MAIN_MODULE_PATH)

        try:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.benchmark"
            )
        except ImportError:
            real_libero_benchmark = importlib.import_module(
                f"{LIBERO_PKG_NAME}.benchmark"
            )

        try:
            real_libero_envs = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.envs"
            )
        except ImportError:
            real_libero_envs = importlib.import_module(f"{LIBERO_PKG_NAME}.envs")

        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs
    except ImportError as e:
        print(
            f"[Main Process Routing Error] Failed to import '{LIBERO_MAIN_MODULE_PATH}'. Error: {e}"
        )

if libero_type == "pro":
    import liberopro.liberopro as _libero_core
    from liberopro.liberopro.benchmark import Benchmark
elif libero_type == "plus":
    import liberoplus.liberoplus as _libero_core
    from liberoplus.liberoplus.benchmark import Benchmark
else:
    import libero.libero as _libero_core
    from libero.libero.benchmark import Benchmark

# Must run before any benchmark lookup: get_task_init_states() reads the cached
# config, so repointing it later (e.g. when resolving bddl_files) is too late.
_repoint_libero_config(_libero_core)


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info

        # Optional exact LIBERO-Plus category evaluation.  When configured, the
        # official task-classification manifest takes precedence over the
        # legacy suffix-based task selection below.
        self.plus_category = cfg.get(
            "plus_category", os.environ.get("LIBERO_CATEGORY", None)
        )
        self.plus_classification_path = cfg.get(
            "plus_classification_path",
            os.environ.get("LIBERO_CLASSIFICATION_PATH", None),
        )
        self.plus_strict_manifest = bool(cfg.get("plus_strict_manifest", True))
        self.plus_expected_num_tasks = cfg.get("plus_expected_num_tasks", None)
        self._plus_manifest_by_task = None
        self._plus_manifest_entries = None

        if seed_offset == 0:
            self._log_evaluation_mode()
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.task_id_filter = cfg.get("task_id_filter", None)
        if self.task_id_filter is not None:
            self.task_id_filter = list(self.task_id_filter)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self.is_eval = cfg.get("is_eval", False)

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_suite: Benchmark = get_benchmark_overridden(cfg.task_suite_name)()

        if get_libero_type() == "plus" and (
            self.plus_category is not None
            or self.plus_classification_path is not None
        ):
            if not self.plus_category or not self.plus_classification_path:
                raise ValueError(
                    "Exact LIBERO-Plus category evaluation requires both "
                    "cfg.plus_category and cfg.plus_classification_path "
                    "(or LIBERO_CATEGORY and LIBERO_CLASSIFICATION_PATH)."
                )
            self._load_plus_category_manifest()

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        if self.is_eval:
            pool = self.reset_state_ids_all[self.seed_offset]
            self._eval_reset_pool = pool[pool >= 0].copy()
        else:
            self._eval_reset_pool = np.array([], dtype=np.int64)
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self.use_step_penalty = getattr(cfg, "use_step_penalty", False)

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None
        self.skip_intermediate_renders = bool(
            OmegaConf.select(cfg, "skip_intermediate_renders", default=False)
        )

    def _log_evaluation_mode(self):
        """Log the LIBERO evaluation mode banner (rank 0 env worker only)."""
        libero_type = get_libero_type()
        if libero_type == "pro":
            perturbation = os.environ.get("LIBERO_PERTURBATION", "all")
            logger.info(f"Evaluation Mode: LIBERO-PRO | Perturbation: {perturbation}")
        elif libero_type == "plus":
            if self.plus_category and self.plus_classification_path:
                logger.info(
                    "Evaluation Mode: LIBERO-PLUS | Category manifest: %s | "
                    "Suite: %s",
                    self.plus_category,
                    self.cfg.task_suite_name,
                )
            else:
                suffix = os.environ.get("LIBERO_SUFFIX", "all")
                logger.info(f"Evaluation Mode: LIBERO-PLUS | Suffix: {suffix}")
        else:
            logger.info("Evaluation Mode: Standard LIBERO")

    @property
    def _uses_plus_category_manifest(self) -> bool:
        return self._plus_manifest_by_task is not None

    def _load_plus_category_manifest(self) -> None:
        """Load one official LIBERO-Plus category and map it to suite tasks.

        The official JSON has the form ``{suite_name: [task records...]}``.
        Each selected BDDL becomes one trial of its matching LIBERO-Plus task,
        allowing the existing ordered evaluation pool to enumerate every
        category item exactly once.
        """
        classification_path = os.path.abspath(
            os.path.expanduser(str(self.plus_classification_path))
        )
        if not os.path.isfile(classification_path):
            raise FileNotFoundError(
                "LIBERO-Plus task classification file does not exist: "
                f"{classification_path}"
            )

        with open(classification_path, "r", encoding="utf-8") as f:
            classification = json.load(f)

        suite_name = str(self.cfg.task_suite_name).lower()
        if suite_name not in classification:
            available = ", ".join(sorted(classification))
            raise KeyError(
                f"Suite {suite_name!r} is absent from {classification_path}. "
                f"Available suites: {available}"
            )

        requested_category = str(self.plus_category).strip()
        category_lookup = {
            str(entry.get("category", "")).strip().casefold(): str(
                entry.get("category", "")
            ).strip()
            for entry in classification[suite_name]
            if entry.get("category")
        }
        canonical_category = category_lookup.get(requested_category.casefold())
        if canonical_category is None:
            available = ", ".join(sorted(set(category_lookup.values())))
            raise KeyError(
                f"Category {requested_category!r} is absent from suite "
                f"{suite_name!r}. Available categories: {available}"
            )

        import liberoplus.liberoplus as l_plus

        _repoint_libero_config(l_plus)
        bddl_root = os.path.abspath(l_plus.get_libero_path("bddl_files"))
        suite_bddl_root = os.path.join(bddl_root, suite_name)
        if not os.path.isdir(suite_bddl_root):
            raise FileNotFoundError(
                f"LIBERO-Plus BDDL suite directory does not exist: "
                f"{suite_bddl_root}"
            )

        suite_tasks_by_name = {}
        for task_id in range(self.task_suite.get_num_tasks()):
            task = self.task_suite.get_task(task_id)
            task_name = str(getattr(task, "name", "")).strip()
            if not task_name:
                task_name = os.path.splitext(os.path.basename(task.bddl_file))[0]
            if task_name in suite_tasks_by_name:
                raise RuntimeError(
                    f"Duplicate task name {task_name!r} in LIBERO-Plus suite "
                    f"{suite_name!r}."
                )
            suite_tasks_by_name[task_name] = (task_id, task)

        manifest_by_task = {
            task_id: [] for task_id in range(self.task_suite.get_num_tasks())
        }
        missing_bddls = []
        unmatched_names = []
        duplicate_names = []
        seen_names = set()

        selected = [
            entry
            for entry in classification[suite_name]
            if str(entry.get("category", "")).strip().casefold()
            == canonical_category.casefold()
        ]

        for entry in selected:
            raw_name = str(entry.get("name", "")).strip()
            name = raw_name[:-5] if raw_name.endswith(".bddl") else raw_name
            if not name:
                unmatched_names.append(raw_name)
                continue
            if name in seen_names:
                duplicate_names.append(name)
                continue
            seen_names.add(name)

            suite_task = suite_tasks_by_name.get(name)
            if suite_task is None:
                unmatched_names.append(name)
                continue

            matching_task_id, task = suite_task

            bddl_path, source_bddl_path, virtual_bddl = _resolve_plus_bddl_paths(
                bddl_root, task.problem_folder, name
            )
            if not os.path.isfile(source_bddl_path):
                missing_bddls.append(source_bddl_path)
                continue

            record = {
                "classification_id": int(entry["id"]),
                "name": name,
                "category": canonical_category,
                "difficulty_level": entry.get("difficulty_level", None),
                "suite_task_id": matching_task_id,
                "bddl_path": os.path.abspath(bddl_path),
                "source_bddl_path": os.path.abspath(source_bddl_path),
                "virtual_bddl": virtual_bddl,
                "local_trial_id": 0,
            }
            manifest_by_task[matching_task_id].append(record)

        if self.plus_strict_manifest and (
            missing_bddls or unmatched_names or duplicate_names
        ):
            details = []
            if missing_bddls:
                details.append(
                    f"missing_bddls={len(missing_bddls)} "
                    f"examples={missing_bddls[:3]}"
                )
            if unmatched_names:
                details.append(
                    f"unmatched_names={len(unmatched_names)} "
                    f"examples={unmatched_names[:3]}"
                )
            if duplicate_names:
                details.append(
                    f"duplicate_names={len(duplicate_names)} "
                    f"examples={duplicate_names[:3]}"
                )
            raise RuntimeError(
                "Invalid LIBERO-Plus category manifest: " + "; ".join(details)
            )

        multiple_records = {
            task_id: records
            for task_id, records in manifest_by_task.items()
            if len(records) > 1
        }
        if multiple_records:
            examples = {
                task_id: [record["name"] for record in records]
                for task_id, records in list(multiple_records.items())[:3]
            }
            raise RuntimeError(
                "Each LIBERO-Plus suite task must map to at most one official "
                f"classification record; duplicates found: {examples}"
            )

        loaded_count = sum(len(records) for records in manifest_by_task.values())
        expected_count = self.plus_expected_num_tasks
        if expected_count is not None and loaded_count != int(expected_count):
            raise RuntimeError(
                f"LIBERO-Plus category {canonical_category!r} for "
                f"{suite_name!r} loaded {loaded_count} tasks, expected "
                f"{int(expected_count)}."
            )
        if loaded_count == 0:
            raise RuntimeError(
                f"LIBERO-Plus category {canonical_category!r} for "
                f"{suite_name!r} contains no usable BDDL tasks."
            )

        self.plus_category = canonical_category
        self.plus_classification_path = classification_path
        self._plus_manifest_by_task = manifest_by_task
        selected_suite_task_ids = [
            task_id for task_id, records in manifest_by_task.items() if records
        ]
        virtual_count = sum(
            int(record["virtual_bddl"])
            for records in manifest_by_task.values()
            for record in records
        )
        logger.info(
            "[LIBERO-PLUS manifest] loaded suite=%s category=%r tasks=%s "
            "suite_task_id_range=%s..%s virtual_tasks=%s source=%s",
            suite_name,
            canonical_category,
            loaded_count,
            min(selected_suite_task_ids),
            max(selected_suite_task_ids),
            virtual_count,
            classification_path,
        )

    def _get_plus_manifest_entry(self, task_id: int, trial_id: int) -> dict:
        if not self._uses_plus_category_manifest:
            raise RuntimeError("LIBERO-Plus category manifest is not enabled.")
        records = self._plus_manifest_by_task[int(task_id)]
        if trial_id < 0 or trial_id >= len(records):
            raise IndexError(
                f"Invalid LIBERO-Plus local trial {trial_id} for base task "
                f"{task_id}; valid range is [0, {len(records) - 1}]."
            )
        return records[int(trial_id)]

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        current_type_val = get_libero_type()

        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param, _type_val=current_type_val):
                os.environ["LIBERO_TYPE"] = _type_val
                seed = param.pop("seed")

                if _type_val in ["pro", "plus"]:
                    sys.path[:] = [p for p in sys.path if "opt/libero" not in p]

                    pkg_name = f"libero{_type_val}"
                    core_name = f"{pkg_name}.{pkg_name}"

                    try:
                        real_pkg = importlib.import_module(pkg_name)
                        real_core = importlib.import_module(core_name)
                        real_bench = importlib.import_module(f"{core_name}.benchmark")
                        real_envs = importlib.import_module(f"{core_name}.envs")

                        sys.modules["libero"] = real_pkg
                        sys.modules["libero.libero"] = real_core
                        sys.modules["libero.libero.benchmark"] = real_bench
                        sys.modules["libero.libero.envs"] = real_envs

                        loaded_path = os.path.dirname(real_core.__file__)
                        os.environ["LIBERO_ASSET_ROOT"] = os.path.join(
                            loaded_path, "assets"
                        )
                        os.environ["LIBERO_BDDL_PATH"] = os.path.join(
                            loaded_path, "bddl_files"
                        )
                        os.environ["LIBERO_INIT_STATES_PATH"] = os.path.join(
                            loaded_path, "init_files"
                        )

                        WorkerEnv = real_envs.OffScreenRenderEnv

                    except ImportError as e:
                        print(f"[Worker Env Error] {e}")
                        raise e
                else:
                    from libero.libero.envs import OffScreenRenderEnv as WorkerEnv

                env = WorkerEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        raw_suffix = os.environ.get(
            "LIBERO_SUFFIX",
            os.environ.get(
                "LIBERO_PERTURBATION",
                self.cfg.get("perturbation_suffix", None)
                if hasattr(self.cfg, "get")
                else None,
            ),
        )
        if variant == "pro":
            import liberopro.liberopro as l_pro

            _repoint_libero_config(l_pro)
            bddl_root = l_pro.get_libero_path("bddl_files")
        elif variant == "plus":
            import liberoplus.liberoplus as l_plus

            _repoint_libero_config(l_plus)
            bddl_root = l_plus.get_libero_path("bddl_files")
        else:
            import libero.libero as l_base

            _repoint_libero_config(l_base)
            bddl_root = l_base.get_libero_path("bddl_files")

        suite_name = self.cfg.task_suite_name.lower()
        suite_keyword = suite_name.replace("libero_", "").strip()

        task_descriptions = []
        pert_init_folders = []
        plus_manifest_entries = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(
                    self.task_descriptions[env_id]
                    if hasattr(self, "task_descriptions")
                    else ""
                )
                pert_init_folders.append(
                    self._pert_init_folders[env_id]
                    if hasattr(self, "_pert_init_folders")
                    else ""
                )
                plus_manifest_entries.append(
                    self._plus_manifest_entries[env_id]
                    if self._plus_manifest_entries is not None
                    else None
                )
                continue

            task = self.task_suite.get_task(self.task_ids[env_id])
            folder_name = task.problem_folder
            file_name = task.bddl_file
            original_path = os.path.join(bddl_root, folder_name, file_name)

            final_path = original_path
            plus_manifest_entry = None

            if variant == "plus" and self._uses_plus_category_manifest:
                plus_manifest_entry = self._get_plus_manifest_entry(
                    int(self.task_ids[env_id]), int(self.trial_ids[env_id])
                )
                final_path = plus_manifest_entry["bddl_path"]
                if self.is_eval:
                    logger.info(
                        "[LIBERO-PLUS manifest task] env=%s "
                        "classification_id=%s task_id=%s trial_id=%s "
                        "category=%r bddl=%s",
                        env_id,
                        plus_manifest_entry["classification_id"],
                        self.task_ids[env_id],
                        self.trial_ids[env_id],
                        plus_manifest_entry["category"],
                        final_path,
                    )

            elif variant == "pro":
                pro_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = ["_lan", "_object", "_swap", "_task"]
                if pro_suffix == "all":
                    filter_perts = valid_perts
                elif pro_suffix is not None:
                    # Map bare name (e.g. "task") to directory suffix (e.g. "_task")
                    normalized = (
                        f"_{pro_suffix}"
                        if not pro_suffix.startswith("_")
                        else pro_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    all_sub_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                        and suite_keyword in d
                        and any(d.endswith(pert) for pert in filter_perts)
                    ]

                    core_task_name = file_name.replace(".bddl", "")
                    all_candidates = []

                    for sub_dir in all_sub_dirs:
                        target_dir_path = os.path.join(bddl_root, sub_dir)
                        matches = [
                            os.path.join(target_dir_path, f)
                            for f in os.listdir(target_dir_path)
                            if core_task_name in f and f.endswith(".bddl")
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            elif variant == "plus":
                plus_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None

                valid_perts = [
                    "_light",
                    "_language",
                    "_table",
                    "_add",
                    "_tb",
                    "_sample",
                    "_level",
                ]
                if plus_suffix == "all":
                    filter_perts = valid_perts
                elif plus_suffix is not None:
                    normalized = (
                        f"_{plus_suffix}"
                        if not plus_suffix.startswith("_")
                        else plus_suffix
                    )
                    filter_perts = [normalized] if normalized in valid_perts else []
                else:
                    filter_perts = []

                if filter_perts:
                    clean_name = file_name.replace(".bddl", "")
                    for marker in valid_perts:
                        if marker in clean_name:
                            clean_name = clean_name.split(marker)[0]
                            break

                    suite_pattern = folder_name.replace("_", "").lower()
                    all_dirs = [
                        d
                        for d in os.listdir(bddl_root)
                        if os.path.isdir(os.path.join(bddl_root, d))
                    ]
                    search_dirs = [
                        os.path.join(bddl_root, d)
                        for d in all_dirs
                        if suite_pattern in d.lower().replace("_", "")
                    ]

                    if not search_dirs:
                        search_dirs = [os.path.join(bddl_root, folder_name)]

                    all_candidates = []
                    for target_dir in search_dirs:
                        matches = [
                            f
                            for f in glob.glob(os.path.join(target_dir, "*.bddl"))
                            if clean_name in os.path.basename(f)
                            and any(
                                pert in os.path.basename(f) for pert in filter_perts
                            )
                        ]
                        all_candidates.extend(matches)

                    if all_candidates:
                        all_candidates.sort()
                        if self.is_eval:
                            idx_offset = (
                                list(env_idx).index(env_id) if env_id in env_idx else 0
                            )
                            final_path = all_candidates[
                                (self.seed + idx_offset) % len(all_candidates)
                            ]
                        else:
                            final_path = self._generator.choice(all_candidates)

            plus_manifest_entries.append(plus_manifest_entry)

            env_fn_params.append(
                {
                    **base_env_args,
                    "bddl_file_name": final_path,
                    "seed": self.seed,
                }
            )
            # LIBERO-PRO: use selected BDDL language (not original suite task.language)
            # and remember the perturbation folder for pruned_init loading.
            pert_folder = os.path.basename(os.path.dirname(os.path.abspath(final_path)))
            pert_init_folders.append(pert_folder)
            if variant == "pro" or plus_manifest_entry is not None:
                prompt_bddl_path = (
                    plus_manifest_entry.get("source_bddl_path", final_path)
                    if plus_manifest_entry is not None
                    else final_path
                )
                bddl_lang, bddl_goal = _read_bddl_language_and_goal(
                    prompt_bddl_path
                )
                desc = bddl_lang if bddl_lang else task.language
                task_descriptions.append(desc)
                if self.is_eval and variant == "pro":
                    logger.info(
                        "[LIBERO-PRO lang] env=%s pert_folder=%s "
                        "prompt=%r suite_orig=%r bddl_lang=%r goal=%r bddl=%s",
                        env_id,
                        pert_folder,
                        desc,
                        task.language,
                        bddl_lang,
                        bddl_goal,
                        final_path,
                    )
            else:
                task_descriptions.append(task.language)

        self.task_descriptions = task_descriptions
        self._pert_init_folders = pert_init_folders
        self._plus_manifest_entries = plus_manifest_entries
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        if self._uses_plus_category_manifest:
            for task_id in range(self.task_suite.get_num_tasks()):
                task_num_trials = len(self._plus_manifest_by_task[task_id])
                self.trial_id_bins.append(task_num_trials)
                self.total_num_group_envs += task_num_trials
        else:
            for task_id in range(self.task_suite.get_num_tasks()):
                task_num_trials = len(self.task_suite.get_task_init_states(task_id))
                self.trial_id_bins.append(task_num_trials)
                self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

        if self.task_id_filter is not None:
            num_tasks = len(self.trial_id_bins)
            validated_tids = []
            for tid in self.task_id_filter:
                if not isinstance(tid, (int, np.integer)):
                    raise ValueError(
                        f"task_id_filter must contain ints, got "
                        f"{type(tid).__name__}: {tid}"
                    )
                tid_int = int(tid)
                if tid_int < 0 or tid_int >= num_tasks:
                    raise ValueError(
                        f"task_id {tid_int} in task_id_filter is out of range "
                        f"[0, {num_tasks - 1}]"
                    )
                validated_tids.append(tid_int)
            validated_tids = sorted(set(validated_tids))

            self._valid_reset_state_ids = []
            for tid in validated_tids:
                start = self.cumsum_trial_id_bins[tid - 1] if tid > 0 else 0
                end = self.cumsum_trial_id_bins[tid]
                self._valid_reset_state_ids.extend(range(start, end))
            self._valid_reset_state_ids = np.array(self._valid_reset_state_ids)
        else:
            self._valid_reset_state_ids = None

    def update_reset_state_ids(self):
        if self.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        elif self._valid_reset_state_ids is not None:
            indices = self._generator.integers(
                low=0, high=len(self._valid_reset_state_ids), size=(num_reset_states,)
            )
            reset_state_ids = self._valid_reset_state_ids[indices]
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        if self.is_eval:
            if self._valid_reset_state_ids is not None:
                reset_state_ids = self._valid_reset_state_ids.copy()
            else:
                reset_state_ids = build_interleaved_eval_reset_state_ids(
                    self.trial_id_bins, self.cumsum_trial_id_bins
                )
            return distribute_reset_state_ids_round_robin(
                reset_state_ids, self.total_num_processes
            )

        if self._valid_reset_state_ids is not None:
            reset_state_ids = self._valid_reset_state_ids.copy()
        else:
            reset_state_ids = np.arange(self.total_num_group_envs)

        self._generator_ordered.shuffle(reset_state_ids)

        # Ensure we have enough IDs for all processes by tiling if needed
        if len(reset_state_ids) < self.total_num_processes:
            repeats = (self.total_num_processes // len(reset_state_ids)) + 1
            reset_state_ids = np.tile(reset_state_ids, repeats)

        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            return self.specific_reset_id * np.ones((num_reset_states,), dtype=int)

        if self.is_eval:
            pool = self._eval_reset_pool
            if self.start_idx >= len(pool):
                return np.full((num_reset_states,), -1, dtype=np.int64)
            end = min(self.start_idx + num_reset_states, len(pool))
            n_valid = end - self.start_idx
            result = np.full((num_reset_states,), -1, dtype=np.int64)
            if n_valid > 0:
                result[:n_valid] = pool[self.start_idx : end]
            self.start_idx = end
            return result

        if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
            self.reset_state_ids_all = self.get_reset_state_ids_all()
            self.start_idx = 0
        reset_state_ids = self.reset_state_ids_all[self.seed_offset][
            self.start_idx : self.start_idx + num_reset_states
        ]
        self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        # LIBERO-PRO: load pruned_init from the selected perturbation folder
        # (e.g. libero_object_task/), not the original suite folder.
        if variant == "pro" and getattr(self, "_pert_init_folders", None):
            import liberopro.liberopro as l_pro

            init_root = l_pro.get_libero_path("init_states")
            init_state = []
            for env_id in env_idx:
                task = self.task_suite.get_task(self.task_ids[env_id])
                folder = self._pert_init_folders[env_id] or task.problem_folder
                pert_init_path = os.path.join(init_root, folder, task.init_states_file)
                states = None
                init_path = None
                used_folder = folder
                if os.path.exists(pert_init_path):
                    loaded = torch.load(pert_init_path, weights_only=False)
                    n = len(loaded) if hasattr(loaded, "__len__") else 0
                    if n == 0:
                        if self.is_eval:
                            logger.warning(
                                "[LIBERO-PRO init] empty pruned_init, skip: %s",
                                pert_init_path,
                            )
                    else:
                        states = loaded
                        init_path = pert_init_path
                if states is None:
                    msg = (
                        "[LIBERO-PRO init] perturbation init missing or empty; "
                        "suite fallback is invalid for eval "
                        f"env={env_id} wanted_folder={folder} "
                        f"file={task.init_states_file} path={pert_init_path}"
                    )
                    if self.is_eval:
                        logger.error(msg)
                        raise RuntimeError(msg)
                    states = self.task_suite.get_task_init_states(self.task_ids[env_id])
                    init_path = f"<suite:{task.problem_folder}/{task.init_states_file}>"
                    used_folder = task.problem_folder
                    logger.warning(
                        "%s; falling back to suite init for training: %s n=%s",
                        msg,
                        init_path,
                        len(states),
                    )
                trial = int(self.trial_ids[env_id])
                if trial >= len(states):
                    trial = trial % len(states)
                init_state.append(states[trial])
                if self.is_eval and env_id == env_idx[0]:
                    logger.info(
                        "[LIBERO-PRO init] env=%s folder=%s trial=%s path=%s n=%s",
                        env_id,
                        used_folder,
                        trial,
                        init_path,
                        len(states),
                    )
            return init_state

        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)
        self._task_success_stats: dict[int, dict[str, int]] = {}
        self._eval_seen_trials: set[tuple[int, int]] = set()

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.success_episode_len[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        # Only accumulate returns while not yet succeeded
        self.returns += step_reward * (~self.success_once)
        # Record episode_len at first success
        new_success_mask = terminations & ~self.success_once
        if new_success_mask.any():
            self.success_episode_len[new_success_mask] = self.elapsed_steps[
                new_success_mask
            ]

        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()

        # Use success episode_len for reward if already succeeded, else current elapsed
        episode_len_for_reward = np.where(
            self.success_once, self.success_episode_len, self.elapsed_steps
        )
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_len_for_reward, 1
        )
        if self._uses_plus_category_manifest and self._plus_manifest_entries:
            episode_info["plus_classification_id"] = np.asarray(
                [
                    int(entry["classification_id"]) if entry is not None else -1
                    for entry in self._plus_manifest_entries
                ],
                dtype=np.int64,
            )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        return {
            "full_image": get_libero_image(obs),
            "wrist_image": get_libero_wrist_image(obs),
            "state": np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )

        states = images_and_states["state"]

        suite_name = str(self.cfg.task_suite_name).lower()
        meta_task_names = []
        meta_categories = []
        meta_variant_ids = []
        meta_classification_ids = []
        for env_index, task_id in enumerate(self.task_ids):
            entry = None
            if self._plus_manifest_entries is not None:
                entry = self._plus_manifest_entries[env_index]
            task = self.task_suite.get_task(int(task_id))
            task_name = str(getattr(task, "name", "")).strip()
            if not task_name:
                task_name = os.path.splitext(
                    os.path.basename(task.bddl_file)
                )[0]
            if entry is not None:
                task_name = str(entry["name"])
                category = str(entry["category"])
                variant_id = str(entry["classification_id"])
                classification_id = int(entry["classification_id"])
            else:
                category = str(self.plus_category or "")
                variant_id = task_name
                classification_id = -1
            meta_task_names.append(task_name)
            meta_categories.append(category)
            meta_variant_ids.append(variant_id)
            meta_classification_ids.append(classification_id)

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
            # Metadata stays outside OpenPI transforms and is consumed only by
            # the native Meta-Subspace task-code router.
            "meta_suite_names": [suite_name] * len(self.task_ids),
            "meta_task_ids": torch.as_tensor(
                self.task_ids.copy(), dtype=torch.long
            ),
            "meta_task_names": meta_task_names,
            "meta_perturbation_categories": meta_categories,
            "meta_variant_ids": meta_variant_ids,
            "meta_classification_ids": torch.as_tensor(
                meta_classification_ids, dtype=torch.long
            ),
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            task_changed = self.task_ids[env_id] != task_ids[j]
            trial_changed = self.trial_ids[env_id] != trial_ids[j]
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
            manifest_changed = self._uses_plus_category_manifest and trial_changed
            if task_changed or manifest_changed or not self.is_eval:
                reconfig_env_idx.append(env_id)
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        variant = os.environ.get(
            "LIBERO_TYPE",
            self.cfg.get("libero_variant", "standard")
            if hasattr(self.cfg, "get")
            else "standard",
        )
        if variant != "plus":
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            if self.is_eval:
                self._task_success_stats = {}
                self._eval_seen_trials = set()
                self.start_idx = 0
                pool = self.reset_state_ids_all[self.seed_offset]
                self._eval_reset_pool = pool[pool >= 0].copy()
                self.update_reset_state_ids()
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        # Preserve LIBERO's existing 15-step settling behavior unless a model
        # explicitly requests a different reset wait (FastWAM uses 30).
        num_steps_wait = int(self.cfg.get("num_steps_wait", 15))
        for _ in range(num_steps_wait):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
                zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def get_camera_meta(
        self, camera_name: str = "agentview", height: int = 256, width: int = 256
    ) -> dict:
        """Fetch camera intrinsics/extrinsics and depth planes.

        Returns camera calibration from worker 0's robosuite sim: intrinsic
        matrix, cam-to-world transform, and depth near/far.  The agentview
        camera is fixed in the world, so this is constant per episode.
        """
        return self.env.workers[0].get_camera_meta(
            camera_name=camera_name, height=height, width=width
        )

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Render an arbitrary camera at the requested resolution.

        Returns:
            The rendered RGB image, or a ``(rgb, depth)`` tuple when
            *depth* is True.
        """
        return self.env.workers[0].render_camera(
            camera_name=camera_name,
            height=height,
            width=width,
            depth=depth,
        )

    def step(self, actions=None, auto_reset=True, _skip_obs_wrap=False):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = None if _skip_obs_wrap else self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(terminations)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos, _ = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        rendering_disabled = False
        try:
            for i in range(chunk_size):
                should_render = (not self.skip_intermediate_renders) or (
                    i == chunk_size - 1
                )
                if self.skip_intermediate_renders and i == 0 and chunk_size > 1:
                    self.env.set_camera_rendering(False)
                    rendering_disabled = True
                elif (
                    self.skip_intermediate_renders
                    and i == chunk_size - 1
                    and rendering_disabled
                ):
                    self.env.set_camera_rendering(True)
                    rendering_disabled = False

                actions = chunk_actions[:, i]
                extracted_obs, step_reward, terminations, truncations, infos = (
                    self.step(
                        actions, auto_reset=False, _skip_obs_wrap=not should_render
                    )
                )
                obs_list.append(extracted_obs)
                infos_list.append(infos)

                chunk_rewards.append(step_reward)
                raw_chunk_terminations.append(terminations)
                raw_chunk_truncations.append(truncations)
        finally:
            if rendering_disabled:
                self.env.set_camera_rendering(True)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1], eval_count_mask = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations

            if eval_count_mask is not None:
                eval_count_mask = torch.tensor(
                    eval_count_mask,
                    dtype=torch.bool,
                    device=past_terminations.device,
                )
                chunk_terminations[:, -1] &= eval_count_mask
                chunk_truncations[:, -1] &= eval_count_mask
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        if self.is_eval:
            return self._handle_eval_auto_reset(dones, _final_obs, infos)
        obs, infos = self._handle_train_auto_reset(dones, _final_obs, infos)
        return obs, infos, None

    def _handle_eval_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        count_mask = record_completed_episode_task_stats(
            env_idx,
            final_info,
            self.task_ids,
            self.trial_ids,
            self.num_envs,
            self._eval_seen_trials,
            self._task_success_stats,
        )

        new_reset_state_ids = self._get_ordered_reset_state_ids(len(env_idx))
        valid_mask = new_reset_state_ids >= 0
        env_to_reset = env_idx[valid_mask]
        if len(env_to_reset) > 0:
            self.reset_state_ids[env_to_reset] = new_reset_state_ids[valid_mask]
            obs, infos = self.reset(
                env_idx=env_to_reset,
                reset_state_ids=self.reset_state_ids[env_to_reset],
            )
        else:
            obs = _final_obs
            infos = {}

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool) & count_mask
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos, count_mask

    def _handle_train_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)

        if self.use_fixed_reset_state_ids:
            self.update_reset_state_ids()
            obs, infos = self.reset(
                env_idx=env_idx,
                reset_state_ids=self.reset_state_ids[env_idx],
            )
        else:
            obs, infos = self.reset(env_idx=env_idx, reset_state_ids=None)

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = np.asarray(dones, dtype=bool)
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        step_penalty = -1 if self.use_step_penalty else 0
        termination_bonus = self.cfg.reward_coef * terminations
        reward = step_penalty + termination_bonus

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            return reward
