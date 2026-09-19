"""One prediction contract for every image classifier.

SOFI queries a black box thousands of times and needs nothing from it beyond
class probabilities. The wrapper below reduces any supported estimator to a
single function that receives an array shaped ``(n_instances, height, width,
n_channels)`` and returns an array shaped ``(n_instances, n_classes)`` whose
rows sum to one.

Four families are recognized without importing their libraries at module load,
so neither PyTorch nor TensorFlow becomes a dependency of the package.

1. A scikit-learn style estimator exposing ``predict_proba``, including the
   pipelines that flatten an image into a feature vector.
2. A Keras or TensorFlow model, evaluated through its call interface.
3. A ``torch.nn.Module``, evaluated under ``no_grad`` on the device of its own
   parameters. Convolutional networks, the architectures of ``timm`` and the
   image models of ``transformers`` all arrive through this branch.
4. Anything else that is callable, or whose output method is named through
   ``output_fn``.

A fitted model is always required, since the explanation describes that model
and nothing else. What ``output_fn`` adds is a way of naming the method of the
model that produces the outputs, for the cases where the detection would pick
the wrong one or where no method is exposed under a familiar name.

Two conventions are resolved once, on a probe batch drawn from the training
data, and then held fixed. The output convention states whether the model emits
probabilities or logits, and the channel layout states whether it expects the
channel axis before the two spatial axes, as PyTorch does, or after them, as
Keras does. Both are detected by default and both can be stated explicitly when
the probe is ambiguous.

A model that exposes hard labels alone is rejected at construction time.
Probability degradation is the quantity SOFI measures, so a label is not enough,
and failing early with a readable message is preferable to failing inside the
search.
"""

from __future__ import annotations

import sys
import warnings
from typing import List, Optional, Sequence

import numpy as np

__all__ = ["ClassifierWrapper"]

_TOLERANCE = 1e-3


def _class_lineage(obj) -> List[str]:
    """Fully qualified names of the classes an object inherits from."""
    return [f"{cls.__module__}.{cls.__name__}" for cls in type(obj).__mro__]


def _looks_like_torch(model) -> bool:
    return any(name.startswith("torch.nn.modules") for name in _class_lineage(model))


def _looks_like_keras(model) -> bool:
    lineage = _class_lineage(model)
    return any(
        name.startswith("keras.") or name.startswith("tensorflow.")
        for name in lineage
    )


def _to_numpy(output) -> np.ndarray:
    """Convert whatever a framework returned into a float array.

    The image models of ``transformers`` return a dataclass rather than a
    tensor, and the field holding the class scores is named ``logits``, so it is
    unwrapped here rather than in every caller. A mapping is treated the same
    way, since a traced or exported model often returns one.
    """
    for attribute in ("logits", "prediction_logits"):
        if hasattr(output, attribute):
            output = getattr(output, attribute)
            break
    else:
        if isinstance(output, dict):
            for key in ("logits", "probabilities", "predictions", "output"):
                if key in output:
                    output = output[key]
                    break
    if isinstance(output, (list, tuple)):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach()
    if hasattr(output, "numpy"):
        try:
            output = output.numpy()
        except (RuntimeError, TypeError):
            # a tensor held on an accelerator has to be moved first, and only
            # that case needs it, so an eager TensorFlow tensor is left alone
            output = output.cpu().numpy()
    return np.asarray(output, dtype=float)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _is_probability(matrix: np.ndarray) -> bool:
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return False
    if np.any(matrix < -_TOLERANCE) or np.any(matrix > 1.0 + _TOLERANCE):
        return False
    return bool(np.allclose(matrix.sum(axis=1), 1.0, atol=1e-2))


class ClassifierWrapper:
    """Uniform probability interface over an image classifier.

    Parameters
    ----------
    model : object
        Fitted classifier. PyTorch modules, including those of ``timm`` and of
        ``transformers``, Keras models and scikit-learn style estimators are
        recognized without further declaration.
    output_fn : str or callable, optional
        Method of the model that produces the outputs. A string names an
        attribute of the model, such as ``"predict_proba"`` or ``"forward"``,
        and a callable receives the batch and returns the outputs directly. The
        detection resolves it when the argument is left out, which covers the
        supported families.
    output : {"auto", "proba", "logits"}, default="auto"
        Whether the raw output already holds probabilities. Under ``auto`` the
        probe decides, and a softmax is applied only when the raw output fails
        the simplex test. Stating the convention removes any doubt.
    channels_first : {"auto", True, False}, default="auto"
        Whether the model expects the channel axis before the two spatial axes,
        which is the PyTorch convention, or after them, which is the Keras
        convention. Under ``auto`` the probe tries both orientations.
    classes : sequence, optional
        Class labels in the order of the probability columns, used for
        reporting alone. They are read from the estimator when available.
    device : str, optional
        Device on which a PyTorch module is evaluated. The device of the
        parameters of the module is used by default.
    batch_size : int, optional
        Largest number of images sent to the model in one call. A cumulative
        marginalization curve of many segments, multiplied by the replicas,
        produces batches an image model cannot always hold, so splitting them
        keeps the memory bounded. The whole batch goes at once by default.
    verbose : bool, default=False
        Whether the resolved conventions are announced once at construction.

    Attributes
    ----------
    n_classes : int
        Number of probability columns, read from the probe.
    backend : str
        Name of the family that was detected.
    """

    def __init__(
        self,
        model,
        output_fn=None,
        output: str = "auto",
        channels_first="auto",
        classes: Optional[Sequence] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ):
        if model is None:
            raise TypeError(
                "A fitted model is required, since the explanation describes "
                "that model. Pass the estimator itself, and use output_fn to "
                "name the method that produces its outputs when the automatic "
                "detection would pick the wrong one."
            )
        if output not in ("auto", "proba", "logits"):
            raise ValueError("output must be 'auto', 'proba' or 'logits'.")
        if channels_first not in ("auto", True, False):
            raise ValueError("channels_first must be 'auto', True or False.")
        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer or None.")

        self.model = model
        self.device = device
        self.batch_size = None if batch_size is None else int(batch_size)
        self._output_fn = output_fn
        self._output = output
        self._channels_first = channels_first
        self.verbose = bool(verbose)

        self.backend, self._raw = self._resolve_backend()
        self.classes = list(classes) if classes is not None else self._read_classes()
        self.n_classes: Optional[int] = None
        self.n_calls_ = 0

    # ----------------------------------------------------------- resolution
    def _resolve_backend(self):
        if self._output_fn is not None and not isinstance(self._output_fn, str):
            if not callable(self._output_fn):
                raise TypeError(
                    "output_fn must be the name of a method of the model or a "
                    "callable receiving a batch."
                )
            # a callable takes full responsibility for the conversion, so it is
            # handed the batch exactly as the explainer holds it
            return "declared", lambda batch: self._output_fn(batch)
        if self._output_fn is not None:
            method = getattr(self.model, self._output_fn, None)
            if not callable(method):
                raise TypeError(
                    f"The model exposes no callable attribute named "
                    f"'{self._output_fn}'."
                )
            # a model of a recognized family keeps that family, so the declared
            # method still receives the input in the form its framework expects.
            # Anything else is called with the batch as the explainer holds it
            if not (
                hasattr(self.model, "predict_proba")
                or _looks_like_keras(self.model)
                or _looks_like_torch(self.model)
            ):
                return f"declared:{self._output_fn}", lambda batch: method(batch)
        if hasattr(self.model, "predict_proba"):
            return "sklearn", self._raw_sklearn
        # Keras is tested before PyTorch on purpose. Under the torch backend a
        # Keras model inherits from torch.nn.Module, so the reverse order would
        # route it into the torch branch and bypass its own call convention.
        if _looks_like_keras(self.model):
            return "keras", self._raw_keras
        if _looks_like_torch(self.model):
            return "torch", self._raw_torch
        if callable(self.model):
            return "callable", lambda batch: self.model(batch)
        raise TypeError(
            "The model exposes neither predict_proba nor a call interface, so "
            "SOFI cannot obtain class probabilities from it. Name the method "
            "that produces them through output_fn."
        )

    def _read_classes(self):
        for attribute in ("classes_", "classes"):
            values = getattr(self.model, attribute, None)
            if values is not None:
                try:
                    return list(values)
                except TypeError:  # pragma: no cover - exotic attributes
                    return None
        # the image models of transformers carry their labels in the config
        config = getattr(self.model, "config", None)
        mapping = getattr(config, "id2label", None)
        if isinstance(mapping, dict) and mapping:
            return [mapping[key] for key in sorted(mapping)]
        return None

    def calibrate(self, probe: np.ndarray) -> "ClassifierWrapper":
        """Fix the output convention and the channel layout on a probe batch.

        Parameters
        ----------
        probe : array
            A few images shaped ``(n, height, width, n_channels)``, normally
            drawn from the training data.

        Returns
        -------
        ClassifierWrapper
            The wrapper itself, so the call chains.
        """
        probe = np.asarray(probe, dtype=float)
        if probe.ndim == 3:
            probe = probe[..., None]
        if probe.ndim != 4:
            raise ValueError(
                "The probe must be shaped (n_instances, height, width, "
                "n_channels)."
            )
        if len(probe) == 0:
            raise ValueError(
                "The probe batch is empty, so the conventions of the model "
                "cannot be resolved. Pass training data that holds at least one "
                "instance."
            )
        if not np.all(np.isfinite(probe)):
            raise ValueError(
                "The probe batch holds values that are not finite. SOFI cannot "
                "resolve the conventions of a model on such input, and the "
                "marginalization statistics would be undefined as well."
            )
        probe = probe[: min(2, len(probe))]

        layouts = (
            [True, False] if self._channels_first == "auto" else [self._channels_first]
        )
        errors = []
        for layout in layouts:
            try:
                raw = _to_numpy(self._raw(self._orient(probe, layout)))
            except Exception as error:  # pragma: no cover - depends on user model
                errors.append((layout, error))
                continue
            if raw.ndim == 1:
                raw = raw.reshape(len(probe), -1)
            if raw.ndim != 2 or raw.shape[0] != len(probe):
                errors.append((layout, ValueError(f"output shaped {raw.shape}")))
                continue
            if raw.shape[1] < 2:
                raise ValueError(
                    "The model returned a single column of scores, so no "
                    "alternative class exists and the probability of the "
                    "predicted class can never fall. SOFI needs a classifier "
                    "with at least two classes. A binary network ending in one "
                    "sigmoid unit qualifies once its output is turned into two "
                    "columns, which output_fn can do."
                )
            self._channels_first = layout
            self.n_classes = int(raw.shape[1])
            break
        else:
            detail = "; ".join(
                f"channels_first={layout} gave {type(err).__name__}, {err}"
                for layout, err in errors
            )
            raise ValueError(
                "The model did not return a matrix of class scores for the "
                "probe batch under either channel layout. State the layout "
                "through channels_first, or name the method that produces the "
                "scores through output_fn. Details, " + detail
            )

        # a square image leaves both layouts admissible, so a warning says that
        # the detection could not tell them apart and the choice was arbitrary
        if (
            self._channels_first is True
            and probe.shape[1] == probe.shape[3]
            and len(layouts) > 1
        ):
            warnings.warn(
                "The probe is as tall as it has channels, so both channel "
                "layouts produced a valid output and the first one was kept. "
                "State channels_first explicitly if the model expects the other.",
                stacklevel=2,
            )

        if self._output == "auto":
            self._output = "proba" if _is_probability(raw) else "logits"
            if self._output == "logits" and self.verbose:
                warnings.warn(
                    "The raw output of the model does not lie on the probability "
                    "simplex, so a softmax is applied to every query. Set "
                    "output='proba' if the model already returns probabilities.",
                    stacklevel=2,
                )
        elif self._output == "proba" and not _is_probability(raw):
            warnings.warn(
                "output='proba' was requested and the probe output does not sum "
                "to one across the classes. The degradation curves will be read "
                "on whatever scale the model produces.",
                stacklevel=2,
            )

        if self.classes is None:
            self.classes = list(range(self.n_classes))
        elif len(self.classes) != self.n_classes:
            warnings.warn(
                f"{len(self.classes)} class labels were declared while the model "
                f"returns {self.n_classes} columns of scores. The labels are used "
                "for reporting alone, so the ranking is unaffected, yet a name "
                "may be missing from a figure or a summary.",
                stacklevel=2,
            )
        if self.verbose:
            print(
                f"[sofiimg] backend {self.backend}, {self.n_classes} classes, "
                f"channels_first={self._channels_first}, output={self._output}"
            )
        return self

    # ------------------------------------------------------------- backends
    @staticmethod
    def _orient(batch: np.ndarray, channels_first: bool) -> np.ndarray:
        return np.transpose(batch, (0, 3, 1, 2)) if channels_first else batch

    def _method(self, default_name: str):
        """The method of the model that produces the outputs.

        The declared name wins when one was given, which is what lets a user
        select ``forward`` over ``__call__`` or an estimator method that the
        detection would not have picked, while the conversion of the input stays
        the one its framework expects.
        """
        name = self._output_fn or default_name
        return getattr(self.model, name)

    def _raw_sklearn(self, batch: np.ndarray):
        method = self._method("predict_proba")
        try:
            return method(batch)
        except Exception:
            # an ordinary estimator reads one row per image, so a batch of
            # images is flattened before the second attempt
            return method(batch.reshape(len(batch), -1))

    def _raw_torch(self, batch: np.ndarray):
        import torch

        device = self.device
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except (StopIteration, AttributeError):  # pragma: no cover
                device = "cpu"
        tensor = torch.as_tensor(batch, dtype=torch.float32).to(device)
        was_training = getattr(self.model, "training", False)
        if was_training:
            self.model.eval()
        method = self._method("__call__") if self._output_fn else self.model
        with torch.no_grad():
            output = method(tensor)
        if was_training:  # pragma: no cover - restores the original mode
            self.model.train()
        return output

    def _raw_keras(self, batch: np.ndarray):
        # the call interface is far cheaper than predict on the small batches
        # SOFI issues, and no gradient is ever needed
        import contextlib

        guard = contextlib.nullcontext()
        torch_module = sys.modules.get("torch")
        if torch_module is not None:
            guard = torch_module.no_grad()
        if self._output_fn:
            method = self._method("__call__")
            with guard:
                return method(batch)
        try:
            with guard:
                return self.model(batch, training=False)
        except (TypeError, ValueError):  # pragma: no cover - older signatures
            return self.model.predict(batch, verbose=0)

    # ---------------------------------------------------------- predictions
    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities of a batch of images.

        Parameters
        ----------
        X : array
            Images shaped ``(n, height, width, n_channels)``. A three
            dimensional array is read as a batch of single channel images.

        Returns
        -------
        ndarray
            Matrix shaped ``(n, n_classes)``.
        """
        batch = np.asarray(X, dtype=float)
        if batch.ndim == 3:
            batch = batch[..., None]
        if batch.ndim != 4:
            raise ValueError(
                "predict_proba expects an array shaped (n_instances, height, "
                f"width, n_channels), and the array received has shape "
                f"{batch.shape}."
            )
        if self.n_classes is None:
            self.calibrate(batch)
        if len(batch) == 0:
            return np.empty((0, self.n_classes), dtype=float)

        size = self.batch_size or len(batch)
        chunks = []
        for start in range(0, len(batch), size):
            piece = batch[start : start + size]
            raw = _to_numpy(self._raw(self._orient(piece, self._channels_first)))
            if raw.ndim == 1:
                raw = raw.reshape(len(piece), -1)
            self.n_calls_ += 1
            chunks.append(raw)
        raw = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        if self._output == "logits":
            raw = _softmax(raw)
        return raw

    def predict(self, X) -> np.ndarray:
        """Position of the most probable class for every image.

        Parameters
        ----------
        X : array
            Images shaped ``(n, height, width, n_channels)``, or a three
            dimensional array read as a batch of single channel images.

        Returns
        -------
        ndarray
            Column position of the most probable class of each image.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def class_name(self, position: int) -> str:
        """Readable label of a class position, for the report and the figures.

        Parameters
        ----------
        position : int
            Column of the probability matrix.

        Returns
        -------
        str
            The declared label, or the position itself when none is known.
        """
        if self.classes is None or position >= len(self.classes):
            return str(position)
        return str(self.classes[position])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ClassifierWrapper(backend={self.backend}, n_classes={self.n_classes}, "
            f"output={self._output}, channels_first={self._channels_first})"
        )
