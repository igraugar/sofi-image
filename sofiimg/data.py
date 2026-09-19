"""Loading of image classification datasets from folders of files.

The loader reads the layout every image library agrees on, namely one directory
per split holding one directory per class, and returns the pixels exactly as
they sit on disk. No resizing, no scaling and no normalization take place, since
the preprocessing belongs to the model rather than to the explainer, and the
same preprocessing has to be applied when the model is built and when an
explanation is produced. Performing it once in the notebook, outside the
package, is what keeps those two uses in step.

Two sources are tried in turn. A directory given by the user comes first, and
the sample bundled with the package follows, so the demonstration runs without
a network connection.

A dataset may carry a second tree of binary masks laid out identically, one file
per image. Those masks are regions of interest rather than labels, and they are
what confines both the classifier and the explanation to the anatomy that
matters. The bundled sample ships the lung fields of every radiograph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "ImageDataset",
    "load_dataset",
    "load_image_folder",
    "bundled_path",
]

_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass
class ImageDataset:
    """Train and test split of one image classification problem.

    Attributes
    ----------
    X_train, X_test : ndarray
        Images shaped ``(n, height, width)`` when they are grayscale and
        ``(n, height, width, n_channels)`` otherwise, in the type they were
        stored in, which is normally an unsigned byte.
    y_train, y_test : ndarray
        Labels, kept as the names of the directories they were read from, so
        that reports stay readable.
    classes : list
        Distinct labels in ascending order, which is the order of the
        probability columns of a classifier trained on ``y_train``.
    name : str
        Name of the problem, used in the summary line.
    source : str
        Where the images came from, either ``"bundled"`` or ``"local"``.
    files : dict
        Paths the images were read from, one list per split, which is what lets
        a figure show the original file beside its preprocessed counterpart.
    masks_train, masks_test : ndarray, optional
        Boolean regions of interest, one per image and aligned with the splits,
        when the dataset carries them. They are what restricts a classifier and
        an explanation to the anatomy of interest.
    """

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    classes: List = field(default_factory=list)
    source: str = "bundled"
    files: dict = field(default_factory=dict)
    masks_train: Optional[np.ndarray] = None
    masks_test: Optional[np.ndarray] = None

    @property
    def has_masks(self) -> bool:
        """Whether regions of interest were found beside the images."""
        return self.masks_train is not None

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def height(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def width(self) -> int:
        return int(self.X_train.shape[2])

    @property
    def n_channels(self) -> int:
        return 1 if self.X_train.ndim == 3 else int(self.X_train.shape[3])

    def codes(self, y) -> np.ndarray:
        """Positions of labels among the classes, as a classifier orders them.

        Parameters
        ----------
        y : array
            Labels drawn from the classes of the dataset.

        Returns
        -------
        ndarray
            Their column positions, which is what a classifier predicts.
        """
        lookup = {str(value): position for position, value in enumerate(self.classes)}
        return np.array([lookup[str(value)] for value in np.asarray(y)], dtype=int)

    def summary(self) -> str:
        return (
            f"{self.name}, {self.n_classes} classes, "
            f"{len(self.X_train)} train and {len(self.X_test)} test images, "
            f"{self.height} by {self.width} pixels, {self.n_channels} channel(s)"
            + (", with lung masks" if self.has_masks else "")
            + f" [{self.source}]"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ImageDataset({self.summary()})"


def bundled_path() -> Optional[Path]:
    """Directory of the sample data shipped with the package, when present."""
    candidate = Path(__file__).resolve().parent / "datasets"
    return candidate if candidate.is_dir() else None


def _read_split(directory: Path, classes: Optional[Sequence] = None, color: bool = False):
    """Read one split, namely one directory holding one directory per class."""
    from PIL import Image

    names = (
        sorted(str(name) for name in classes)
        if classes is not None
        else sorted(child.name for child in directory.iterdir() if child.is_dir())
    )
    if not names:
        raise FileNotFoundError(
            f"The directory {directory} holds no subdirectory, so no class can "
            "be read. The loader expects one directory per class inside each "
            "split."
        )

    images, labels, paths = [], [], []
    for name in names:
        folder = directory / name
        if not folder.is_dir():
            raise FileNotFoundError(f"The class directory {folder} does not exist.")
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in _EXTENSIONS:
                continue
            with Image.open(path) as handle:
                image = handle.convert("RGB" if color else "L")
                images.append(np.asarray(image))
            labels.append(name)
            paths.append(str(path))
    if not images:
        raise FileNotFoundError(
            f"No image file was found under {directory}. Accepted extensions "
            f"are {_EXTENSIONS}."
        )

    shapes = {array.shape for array in images}
    if len(shapes) > 1:
        raise ValueError(
            f"The images under {directory} do not share one shape, and "
            f"{len(shapes)} different ones were found. Resize them before they "
            "are loaded, or read them one by one."
        )
    return np.stack(images), np.array(labels), paths, names


def _read_masks(directory: Path, paths: Sequence[str], root: Path) -> np.ndarray:
    """Read the mask matching every image, by the path it was read from."""
    from PIL import Image

    masks = []
    for path in paths:
        relative = Path(path).relative_to(root)
        candidate = directory / relative
        for suffix in (".png", ".bmp", ".tif", relative.suffix):
            probe = candidate.with_suffix(suffix)
            if probe.exists():
                with Image.open(probe) as handle:
                    masks.append(np.asarray(handle.convert("L")) > 127)
                break
        else:
            raise FileNotFoundError(
                f"No mask was found for {path} under {directory}. The mask tree "
                "must mirror the image tree, one file per image."
            )
    return np.stack(masks)


def load_image_folder(
    directory,
    name: Optional[str] = None,
    train: str = "train",
    test: str = "test",
    classes: Optional[Sequence] = None,
    color: bool = False,
    mask_directory=None,
) -> ImageDataset:
    """Read a dataset laid out as one directory per split and per class.

    Parameters
    ----------
    directory : str or Path
        Root holding the two split directories.
    name : str, optional
        Name of the problem. The name of the directory is used by default.
    train, test : str
        Names of the two split directories.
    classes : sequence, optional
        Class directories to read and the order they are read in. Every
        subdirectory is read in ascending order by default, which is the order
        a classifier trained on the labels will use.
    color : bool, default=False
        Whether the images are read as three channels. A radiograph is
        grayscale, so the default converts to one channel, and a photograph
        needs this turned on.
    mask_directory : str or Path, optional
        Root of a tree of binary masks mirroring the image tree, one file per
        image. The masks are read as regions of interest and returned beside
        the images.

    Returns
    -------
    ImageDataset
        The two splits, the labels, the class names and the masks when a mask
        tree was given.

    Examples
    --------
    >>> data = load_image_folder("~/chest_xray")
    >>> data.X_train.shape
    (1200, 224, 224)
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"The directory {root} does not exist.")
    for split in (train, test):
        if not (root / split).is_dir():
            raise FileNotFoundError(
                f"The split directory {root / split} does not exist. The loader "
                f"expects '{train}' and '{test}' inside the root, each holding "
                "one directory per class."
            )

    X_train, y_train, train_files, names = _read_split(root / train, classes, color)
    X_test, y_test, test_files, _ = _read_split(root / test, names, color)

    masks_train = masks_test = None
    if mask_directory is not None:
        mask_root = Path(mask_directory).expanduser().resolve()
        if not mask_root.is_dir():
            raise FileNotFoundError(f"The mask directory {mask_root} does not exist.")
        masks_train = _read_masks(mask_root, train_files, root)
        masks_test = _read_masks(mask_root, test_files, root)

    return ImageDataset(
        name=name or root.name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classes=list(names),
        source="local",
        files={"train": train_files, "test": test_files},
        masks_train=masks_train,
        masks_test=masks_test,
    )


def load_dataset(
    name: str = "pneumonia",
    directory=None,
    classes: Optional[Sequence] = None,
    color: bool = False,
    masks: bool = True,
    mask_directory=None,
) -> ImageDataset:
    """Load a dataset from a directory, or the sample bundled with the package.

    Parameters
    ----------
    name : str, default="pneumonia"
        Name of the problem. It selects the bundled sample when no directory is
        given, and it names the dataset otherwise.
    directory : str or Path, optional
        Root of the data. The bundled sample is read when it is left out.
    classes : sequence, optional
        Class directories to read and the order they are read in.
    color : bool, default=False
        Whether the images are read as three channels.
    masks : bool, default=True
        Whether a mask tree beside the images is read when one exists. The
        bundled sample carries the lung fields of every radiograph.
    mask_directory : str or Path, optional
        Root of the mask tree, when it does not sit beside the images under the
        name the loader looks for.

    Returns
    -------
    ImageDataset
        The two splits, the labels, the class names and the masks.

    Examples
    --------
    >>> load_dataset("pneumonia")                    # bundled, with lung masks
    >>> load_dataset("my_problem", directory="~/data")
    """
    if directory is not None:
        return load_image_folder(
            directory, name=name, classes=classes, color=color,
            mask_directory=mask_directory,
        )

    root = bundled_path()
    if root is None or not (root / name).is_dir():
        raise FileNotFoundError(
            f"No dataset named '{name}' is bundled with the package. Pass the "
            "directory holding it through the directory argument, laid out as "
            "one directory per split and one per class inside each split."
        )
    bundled_masks = root / f"{name}_masks"
    if mask_directory is None and masks and bundled_masks.is_dir():
        mask_directory = bundled_masks
    dataset = load_image_folder(
        root / name, name=name, classes=classes, color=color,
        mask_directory=mask_directory if masks else None,
    )
    dataset.source = "bundled"
    return dataset
