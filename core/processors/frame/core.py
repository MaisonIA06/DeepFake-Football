import sys
import importlib
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import core.globals

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'core.processors.frame.{frame_processor}')
        return frame_processor_module
    except ImportError as e:
        print(f"Frame processor {frame_processor} not found: {e}")
        return None


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            if frame_processor_module:
                FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    return FRAME_PROCESSORS_MODULES


def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
    with ThreadPoolExecutor(max_workers=core.globals.execution_threads) as executor:
        futures = []
        for path in temp_frame_paths:
            future = executor.submit(process_frames, source_path, [path], progress)
            futures.append(future)
        for future in futures:
            future.result()


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        progress.set_postfix({'execution_providers': core.globals.execution_providers, 'execution_threads': core.globals.execution_threads, 'max_memory': core.globals.max_memory})
        multi_process_frame(source_path, frame_paths, process_frames, progress)
