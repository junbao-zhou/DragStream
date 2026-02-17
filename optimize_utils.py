import json
from pathlib import Path
from attr import dataclass
import numpy as np
import torch

from tensor_utils import (
    calculate_angle_from_points,
    read_mask_from_file,
    save_mask_to_file,
)


@dataclass
class Trajectory:
    original_trajectory: dict[str, bool | list[torch.Tensor]] = None
    """
    trajectory is dict, keys include 'is_rotation', 'points', if translation also has 'control_points'
    """
    block_trajectories: list[dict[str, bool | list[torch.Tensor]]] = []
    """block_num x trajectory
    trajectory has keys 'is_rotation' 'deltas' 'start_point'
    if is_rotation: trajectory also has 'rotation_center'
    """
    mask: np.ndarray = None
    """
    target mask for the trajectory
    """

    def __init__(
        self,
        original_trajectory: dict[str, bool | list[torch.Tensor]] = None,
        mask: np.ndarray = None,
    ):
        self.original_trajectory = original_trajectory
        self.mask = mask
        if original_trajectory is not None:
            self.block_trajectories = self.original_to_block_trajectories(original_trajectory)
        else:
            self.block_trajectories = []

    @staticmethod
    def original_to_block_trajectories(
        original_trajectory: dict[str, bool | list[torch.Tensor]],
        block_length: int = 3,
    ) -> list[dict[str, bool | list[torch.Tensor]]]:
        """Convert an original trajectory (with 'points') into per-block trajectories (with 'deltas').

        For translation:
            deltas[i] = points[i+1] - points[0]   (displacement from start)
            Each block gets `block_length` consecutive deltas.

        For rotation:
            points[0] is the rotation center.
            deltas[i] = angle(center, points[1], points[i+2])
            Each block gets `block_length` consecutive deltas,
            plus 'rotation_center' and 'start_point'.
        """
        is_rotation = original_trajectory.get("is_rotation", False)
        points = original_trajectory.get("points", [])

        if is_rotation:
            # points[0] = rotation center, points[1] = start arm, points[2:] = subsequent arms
            if len(points) < 2:
                return []
            rotation_center = points[0]
            start_point = points[1]
            deltas = [
                calculate_angle_from_points(
                    rotation_center,
                    start_point,
                    point,
                )
                for point in points[2:]
            ]
        else:
            # Translation: points[0] = start, points[1:] = subsequent positions
            if len(points) < 1:
                return []
            start_point = points[0]
            deltas = [torch.Tensor(point) - torch.Tensor(start_point) for point in points[1:]]

        block_trajectories = []
        for i in range(0, len(deltas), block_length):
            block_traj = {
                "is_rotation": is_rotation,
                "deltas": deltas[i : i + block_length],
                "start_point": start_point,
            }
            if is_rotation:
                block_traj["rotation_center"] = rotation_center
            block_trajectories.append(block_traj)
        return block_trajectories

    def set_original_trajectory(
        self,
        original_trajectory: dict[str, bool | list[torch.Tensor]] = None,
    ):
        self.original_trajectory = original_trajectory
        if original_trajectory is not None:
            self.block_trajectories = self.original_to_block_trajectories(original_trajectory)
        else:
            self.block_trajectories = []

    @staticmethod
    def _serialize_value(
        v,
    ):
        """Recursively serialize a value to JSON-compatible types."""
        if isinstance(v, torch.Tensor):
            return v.tolist()
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, dict):
            return {k: Trajectory._serialize_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [Trajectory._serialize_value(item) for item in v]
        else:
            return v

    def to_dict(
        self,
        mask_filename: str = None,
    ) -> dict:
        """Convert the Trajectory to a JSON-serializable dictionary.

        Args:
            mask_filename: If provided, store this filename instead of the mask array.
        """
        result = {}

        if self.original_trajectory is not None:
            result["original_trajectory"] = self._serialize_value(self.original_trajectory)
        else:
            result["original_trajectory"] = None

        result["block_trajectories"] = self._serialize_value(self.block_trajectories)

        if mask_filename is not None:
            result["mask_file"] = mask_filename

        return result

    def save_mask(
        self,
        save_path: Path,
    ) -> None:
        """Save the mask as a PNG image."""
        if self.mask is not None:
            save_mask_to_file(self.mask, save_path)

    @staticmethod
    def load(
        data: dict,
        save_dir: Path,
    ) -> "Trajectory":
        """Load a Trajectory from a dictionary and directory."""
        traj = Trajectory()
        traj.original_trajectory = data.get("original_trajectory", None)
        traj.block_trajectories = data.get("block_trajectories", [])
        mask_file = data.get("mask_file", None)
        if mask_file is not None:
            traj.mask = read_mask_from_file(save_dir / mask_file)
        return traj


@dataclass
class MultiTrajectory:
    block_number: int = 1
    prompt: str = ""
    drag_or_animation_select: str = "Drag"
    trajectories: list[Trajectory] = []
    """
    multiple trajectories for a single prompt, each trajectory has its own mask
    """
    movable_mask: np.ndarray = None
    """
    the movable area mask for the whole image
    """

    def save(
        self,
        save_dir: str | Path,
        prefix: str = "multi_traj",
    ) -> Path:
        """Save the MultiTrajectory to a directory.

        Masks are saved as PNG images, and metadata is saved as a JSON file.

        Args:
            save_dir: Directory to save files into.
            prefix: Filename prefix for all saved files.

        Returns:
            Path to the saved JSON file.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "block_number": self.block_number,
            "prompt": self.prompt,
            "drag_or_animation_select": self.drag_or_animation_select,
        }

        # Save movable_mask
        if self.movable_mask is not None:
            movable_mask_filename = f"{prefix}_movable_mask.png"
            save_mask_to_file(self.movable_mask, save_dir / movable_mask_filename)
            result["movable_area_mask_file"] = movable_mask_filename
        else:
            result["movable_area_mask_file"] = None

        # Save each trajectory and its mask
        traj_dicts = []
        if self.trajectories is not None:
            for i, traj in enumerate(self.trajectories):
                mask_filename = None
                if traj.mask is not None:
                    mask_filename = f"{prefix}_traj_{i}_mask.png"
                    traj.save_mask(save_dir / mask_filename)
                traj_dicts.append(traj.to_dict(mask_filename=mask_filename))
        result["trajectories"] = traj_dicts

        # Write JSON
        json_path = save_dir / f"{prefix}_trajectory.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        return json_path

    @staticmethod
    def load(
        save_dir: str | Path,
        prefix: str = "multi_traj",
    ) -> "MultiTrajectory":
        """Load a MultiTrajectory from a directory.

        Args:
            save_dir: Directory containing the saved files.
            prefix: Filename prefix used when saving.

        Returns:
            The loaded MultiTrajectory instance.
        """
        save_dir = Path(save_dir)
        json_path = save_dir / f"{prefix}_trajectory.json"

        with open(json_path, "r") as f:
            data = json.load(f)

        mt = MultiTrajectory()
        mt.block_number = data.get("block_number", 1)
        mt.prompt = data.get("prompt", "")
        mt.drag_or_animation_select = data.get("drag_or_animation_select", "Drag")
        # Load movable_mask
        movable_file = data.get("movable_area_mask_file", None)
        if movable_file is not None:
            mt.movable_mask = read_mask_from_file(save_dir / movable_file)

        # Load trajectories
        mt.trajectories = []
        for traj_data in data.get("trajectories", []):
            mt.trajectories.append(Trajectory.load(traj_data, save_dir))

        return mt


def transpose_dict_2d(d):
    """Transpose a 2D dict: dict[key1][key2] -> dict[key2][key1]."""
    result = {}
    for key1, inner in d.items():
        for key2, item in inner.items():
            result.setdefault(key2, {})[key1] = item
    return result
