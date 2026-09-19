# Sparseness Optimized Feature Importance for Image Classification

Sparseness Optimized Feature Importance (SOFI) is a model agnostic, declarative post hoc explainer. An explanation takes the form of a ranking of image segments, and its quality is the degradation score obtained after cumulative marginalization. This package is the image counterpart of `sofits`. It supports grayscale and color problems, confines both the ranking and the marginalization to a region of interest when one is given, and works with any classifier that yields class probabilities, including every convolutional network, vision transformer and hybrid backbone published through `torchvision`, `timm`, `transformers` and Keras.

## Installation

```bash
git clone https://github.com/<user>/sofi-image.git
cd sofi-image
pip install -e .
```

Python 3.9 or later is required. Everything the explainer and its figures need is installed with it, including `scikit-image` for the SLIC segmentation, `scipy`, `pillow`, `matplotlib`, `seaborn`, `pandas`, `tqdm` and `imageio-ffmpeg` for the animation. Neither PyTorch nor TensorFlow is a dependency, since the package works with both and imports neither until the model given to it requires one.

Two extras cover the frameworks. `pip install -e ".[demo]"` adds TensorFlow, Keras and a notebook kernel, which is what `SOFI_demo.ipynb` needs. `pip install -e ".[torch]"` adds PyTorch together with `torchvision`, `timm` and `transformers`.

## Quick start

Three objects carry ordinary use. `load_dataset` reads images from disk, `SOFIExplainer` explains one of them, and the `SOFIExplanation` it returns draws itself. A fourth, `Experiment`, is reserved for batch work and sensitivity studies.

```python
from sofiimg import SOFIExplainer, load_dataset

data = load_dataset("pneumonia")
X_train, X_test = preprocess(data.X_train), preprocess(data.X_test)   # your own pipeline

explainer = SOFIExplainer(model, X_train, y_train, classes=data.classes, random_state=42)
explanation = explainer.explain(X_test[0], display=data.X_test[0])
print(explanation.summary())

explanation.plot_explanation()      # the segments beside the two curves
explanation.plot_segmentation()     # the partition on the image
explanation.plot_marginalization()  # the image along the cumulative removal
explanation.plot_animation()        # the same process animated, shown in the cell
```

`Experiment` covers the research half, namely explaining a batch and measuring how a setting changes the result. Each study receives the image it runs on and returns a table carrying the degradation score, the fall of the response, its recovery, the sparsity rate and the cost, and draws one figure per setting.

```python
from sofiimg import Experiment

study = Experiment(explainer, random_state=42)
study.batch(X_test, y_test, masks=rois)                 # explain a collection
study.segment_counts(image, [10, 20, 40, 80])           # how fine the partition should be
study.compactness(image, [0.05, 0.2, 2.0])              # how closely segments follow the content
study.operators(image, ["mean", "blur", "road", "lowneg"])   # what neutralizing a segment means
study.rankings(image, saliency={"Input gradient": saliency})  # against other attribution methods
```

`SOFI_demo.ipynb` shows the basic pipeline in full and then each of these studies.

## The degradation score

Two curves are derived from a ranking. The MoRF curve marginalizes the most relevant segments first and should collapse immediately, since a sparse explanation concentrates the response of the model in a handful of segments. The LeRF curve marginalizes the least relevant segments first and should stay flat for as long as possible, since the segments declared irrelevant must indeed be the ones whose removal leaves the response untouched. Both curves share their first point, the unperturbed response, and their last point, the response once every segment has been neutralized. The degradation score is the area between the two curves, and the search maximizes it. A single curve cannot capture both properties, because the MoRF branch alone rewards sparsity while saying nothing about the tail of the ranking, and the LeRF branch alone rewards correctness at the tail while saying nothing about the head.

The search works on the raw response of the model and on nothing else, as in segment perturbation, namely the probability the model assigns to the class it predicted before any perturbation. Ground truth never enters, which keeps explanations independent of the error of the model.

Normalization never enters the optimization, since a scale that depends on the ranking being scored would change what is being maximized. It is applied afterwards, for reporting and for drawing, and it follows the convention of the perturbation curve literature. The response of the unperturbed image is placed at one and the response of the fully marginalized image at zero. Both anchors belong to the model and the instance rather than to a ranking, so every ranking scored on the same image shares them, the reported score is the optimized score divided by one positive constant, and no ordering can change. Reported curves therefore start at one and stay inside the unit interval, so figures from different images, models and problems can be read on the same axis.

The anchors are read off a reference sweep that marginalizes every segment on its own and then follows the resulting greedy ordering cumulatively, which also produces the fully marginalized response. Should the response climb substantially above the unperturbed one, the scale widens to the extremes the sweep observed and a warning names the cause, so that a curve is never flattened against a false ceiling. That warning is worth reading rather than silencing, because on images it usually means the substitution left a patch the model reads with confidence, and `marginalization="auto"` is the direct remedy.

## Two diagnostics beside the score

A degradation score says how the two curves separate. It says nothing about whether the model moved at all, and nothing about where the curve ends. On images both omissions matter, and every explanation therefore carries two further figures.

`explanation.drop` is how far the response falls at its lowest, as a fraction of where it started. An operator that leaves the response within a thousandth of its starting value still produces a curve, because reporting maps the highest response observed onto one and the lowest onto zero, and that mapping stretches a range of a few thousandths across the whole unit interval. The score is then large and describes numerical noise. On the bundled problem the two highest scoring operators, at 0.81 and 0.77, both move the model by one thousandth and never change the predicted class.

`explanation.surge` is how far the response climbs back once every segment has been marginalized, on the same scale. The last point of a curve is the response to an image in which everything has been replaced, and that image is uninformative only when the model does not read it as a member of the explained class. A flat field or a uniformly blurred image is not a radiograph with pieces removed, it is a different kind of object, and the response there owes nothing to the evidence taken away. The constants recover almost completely on the bundled problem, ending within five percent of where they began.

These two figures are also how a reader decides which images are worth explaining at all, since a curve that never moves or that ends where it began cannot be presented as progressive evidence removal whatever the ranking is. The package reports them and leaves the choice to the analyst. `marginalization="auto"` screens on both before comparing any area. A candidate must fall by at least `min_drop` and end below `surge_tolerance`, and the score is maximized only among what survives. A small surge stays admissible on purpose, since marginalizing a segment that holds nothing but noise can genuinely help the classifier, so a curve is not required to be monotone, only to end near its floor. When nothing survives, the failure is named in a warning rather than hidden behind a number.

Two operators are built for this. `stable_constant` keeps the substitution a constant and chooses which one on the terminal criterion alone, assembling a grid of candidates spanning the training range, building the fully marginalized image for each, and keeping the one whose terminal response is lowest. That is a single batched query of a dozen images, paid once per image and independent of the segment count, and it warns when no constant avoids the surge. `lowneg` draws a reference at random from the training images the model scores lowest for the explained class, holds it fixed, and replaces each segment with the spatially corresponding patch, so every substitution is a real piece of a real image the model already reads as belonging elsewhere and the terminal state is the reference itself.

## The sparsity rate

The sparsity rate is the number of segments that must be marginalized before the predicted class changes, divided by the segment count, so lower values denote sparser explanations. A ranking whose cumulative marginalization never changes the class receives a rate of one and a position of `None`, and the two outcomes are reported separately rather than conflated. The same measure applied to the reversed ranking is reported by `lerf_sparsity_rate`, and the distance between the two is what the ordering buys. The figure marks both events with a star, one on each curve.

## Declaring the segments

Segments are the unit of interpretation and they belong to the user. A radiologist reads a chest image in terms of lung fields, costophrenic angles and the mediastinum, so an explanation phrased in those terms is meaningful while one phrased in pixels is not. Seven ways of declaring them are available, and the demo devotes a part to comparing them.

| `segmentation` | What it does | Key arguments |
| --- | --- | --- |
| `"slic"` | Simple linear iterative clustering, the default | `n_segments`, `compactness`, `sigma`, `mask` |
| `"mask"` | An integer label map supplied by the user | `mask` |
| an array | Read directly as the label map | none |

SLIC is the only procedure the package offers. The alternatives of `scikit-image` and the rectangular patch grid were removed rather than kept as options, since a second vocabulary that no one uses is a maintenance cost and a distraction from the two choices that matter, namely the segment count and the region of interest. Segments computed by another library still enter, as a label map.

```python
from sofiimg import SOFIExplainer, segmentation_from_mask

explainer = SOFIExplainer(model, X_train, segmentation="slic",
                          segmentation_params={"n_segments": 40, "compactness": 0.2})

# segments computed by another library, as a label map
explainer = SOFIExplainer(model, X_train, segmentation=segmentation_from_mask(labels))
```

The segment count is the most consequential setting and more is not simply better. A coarse partition names a large share of the anatomy, because the smallest thing it can point at is already large, and refining it shrinks the area the explanation claims. Against that, the search space is the factorial of the count, so the same budget explores an ever smaller fraction of it and the ranking becomes less trustworthy, while the queries and the runtime grow. The degradation score does none of the work one might expect here: across a sweep from nine to seventy-five segments on the bundled problem it moves between 0.47 and 0.55 with no trend, so it neither rewards nor penalizes a finer partition and cannot be used to choose one. Scores obtained under different partitions are not comparable, since each is an area over a different number of steps. The sparsity rate stays comparable, since it is a fraction of the partition rather than a count, and it falls from 0.44 to 0.12 across the same sweep.

SLIC clusters pixels in a space that joins color to position, so a segment is homogeneous in appearance and compact in the plane. Both properties matter. Homogeneity makes a segment a plausible unit of meaning rather than an arbitrary window, and compactness keeps it small enough to point at. The procedure returns a count close to the one requested rather than equal to it, since clusters are seeded on a grid and small ones are absorbed afterwards, and a large shortfall is announced.

`compactness` weighs the spatial term against the color term. The default of `0.2` suits the unit interval that the segmentation step maps the values onto internally, so the same setting works whether the array handed to the explainer is scaled or standardized. A value near ten belongs to the CIELAB space that `convert2lab=True` produces, where the color term spans a hundred units rather than one. The default of `scikit-image` assumes that space, and using it on a grayscale radiograph collapses the partition into a plain square grid.

Negative entries of a label map stay outside every segment and are never marginalized, which is how a background is excluded from an explanation altogether. `slic_segmentation` accepts a boolean `mask` for the same purpose.

Segments apply to one image at a time, since the content of one image says nothing about the content of another. Rankings from different images therefore describe different vocabularies and cannot be pooled unless they share a partition, which is what `aggregate_explanations` checks before it summarizes a collection.

## Restricting an explanation to a region of interest

A time series has to be read whole, since every stretch of it is part of the signal. An image does not. A radiographic classifier trained on unrestricted images can settle on positioning cues, on soft tissue beyond the thorax, or on a border artifact, and an explainer applied to such a model reports those signals faithfully. The explanation is then worthless to a clinician however faithful it is, because the evidence it names carries no meaning in the domain.

The package answers this with a `mask` argument, and the demonstration pairs it with a first segmentation stage that produces the mask. The two stages divide the problem cleanly.

| Stage | What it divides | Who owns it |
| --- | --- | --- |
| Region of interest | the image into the anatomy of interest and everything else | the data and the model, so it belongs outside the package |
| Segments to rank | that anatomy into segments | the explainer, through `mask` |

The first stage restricts what the classifier may look at, by multiplying the image with a binary mask before the model ever sees it. The second restricts what the explanation may talk about. Neither is useful alone, since a model that reads the shoulder cannot be explained honestly inside the lungs, and a model that reads only the lungs still needs its evidence localized within them.

```python
X = preprocess(images, masks)                        # the model sees the anatomy alone
explanation = explainer.explain(X[i], mask=roi[i])  # the ranking stays inside it
```

The mask reaches the explainer through three routes, namely a per-call argument on `explain`, `segment`, `inspect`, `score_ranking` and `build_objective`, the `mask` argument of the constructor when one segment serves every image, and `segmentation_params` when it is more natural to group it with the other segmentation settings. It is read as boolean, so a mask stored as bytes needs no conversion, and it must cover the image it restricts at the same resolution, since a mask computed elsewhere would otherwise name pixels that are not the ones it ranks.

Segments are then grown inside the segment alone and every pixel outside stays out of the partition, with a label of minus one. Such a pixel can never enter a ranking and is never marginalized, so the confinement is structural rather than a filter applied afterwards.

On the bundled problem the effect is stark. The same architecture reaches 0.935 on the whole image and 0.912 on the lung fields, and sixty-five percent of the pixels the unrestricted model relies on lie outside the lungs. The accuracy the first stage costs is the shortcut it removes.

Where masks are not given, the first stage is a segmentation network, typically a U-Net, and nothing downstream changes. The bundled dataset carries expert annotations, so the demonstration loads them instead.

## Marginalizing a segment

Marginalizing a tabular feature is easy, since one statistic of the training column carries no instance level information. An image segment is harder. A flat patch is itself a pattern, and a model may read the sharp contour it leaves behind as evidence for some class, which keeps the marginalized image informative and the curve misleading. Fifteen operators are available in two families, and the right one is problem dependent.

**Class agnostic operators** erase the content of a segment without steering the prediction anywhere. A ranking obtained this way answers which segments the prediction relies on, which is the question the method was designed for. These are the default.

| Name | Substitution |
| --- | --- |
| `"mean"` | The training mean, the default and the closest counterpart of the gray patch of the occlusion literature |
| `"min"`, `"max"`, `"median"`, `"zero"` | Other constants of the training data |
| `"random"` | A fresh uniform draw per segment and per replica |
| `"noise"` | Gaussian noise matched to the training level and dispersion |
| `"segment_mean"` | The average color of the segment itself, which erases texture and keeps brightness |
| `"blur"` | The blurred image restricted to the segment, which introduces no contour, driven by `sigma` |
| `"inpaint"` | A biharmonic reconstruction from the border of the segment, the most faithful and the most expensive |
| `"background"` | The matching pixels of several training images, averaged over the draws |

Every constant reads the statistic of the channel it is writing into rather than a value pooled over all of them. A pipeline that standardizes with the constants of a pretrained backbone leaves the channels centered on different values, so a pooled constant would neutralize one channel while introducing a visible cast in another. The two coincide when the image is grayscale.

`"blur"` is the image counterpart of the `"linear"` operator of the time series package, since both remove the content of a segment while leaving the boundary continuous, and `"linear"` is accepted as an alias so that code written against one package reads the other. `"background"` assumes the images are registered, which radiographs of one protocol are and photographs of arbitrary scenes are not.

**Class directed operators** choose the substitution so that the response falls as fast as possible. They produce sparser explanations and a different reading, since a ranking obtained this way answers which segments move the model away from the current class fastest, which is a counterfactual question rather than a reliance question. Every explanation states which family produced it, in `explanation.interpretation` and in the printed summary, so the two readings are never confused.

| Name | Substitution |
| --- | --- |
| `"stable_constant"` | The constant chosen so that the fully marginalized image is the one the model reads least as the explained class |
| `"lowneg"` | The matching pixels of a reference the model already scores near zero for the explained class |
| `"line_search"` | The constant found by the golden section search of the paper |
| `"admissible_line_search"` | The same search restricted to constants whose flat image is not assigned the explained class, so the substitution is known in advance to carry no evidence for it |
| `"opposite_mean"` | The matching pixels of the mean image of the other classes |
| `"nearest_unlike"` | The matching pixels of the nearest training image of another class |

Two further controls apply to every operator. `n_replicas` averages the response over several substitutions, which approximates the expectation the method is defined on rather than the output at one arbitrary point, at a cost that grows linearly with the sample. `taper` blends the substituted values into the surrounding image across a band of pixels measured inwards from the border of a segment, which removes the sharp contour a flat patch introduces. Convolutional models react strongly to edges, so part of a measured degradation can be a boundary artifact rather than lost evidence, and the default of zero reproduces the published behavior. A band wider than the interior of the smallest segment is announced, since those segments would never be fully neutralized.

No single operator fits every problem. Setting `marginalization="auto"` runs the search once per candidate and keeps the highest degradation score, which treats the choice as a per instance hyperparameter. The candidates are the cheap class agnostic operators, so the reading of the ranking never changes without the user asking, and `explanation.perturbation["trials"]` reports what each of them achieved.

## The noise region and the recovery of fidelity

Once the informative segments are gone, marginalizing what remains often pushes the response back towards its original state. The MoRF curve then climbs after its minimum, and the tail of the ranking that follows that minimum carries no evidence. The segment is reported by `noise_onset` and `noise_features`, and it also appears in the printed summary. Segments inside it receive no importance weight, since the ranking says nothing about them, and the overlay figures leave them bare.

On images the recovery is the rule rather than the exception, and the demonstration devotes a section to measuring it. The last point of a curve is the response to an image in which every segment has been replaced, and such an image is uninformative only when the model does not read it as a member of the explained class. Whether it does depends on the operator. A image replaced everywhere by one constant, or blurred everywhere, is smooth and bright and outside anything the network was trained on, so its response there is arbitrary rather than low, and on the bundled problem it returns almost to the original confidence. An image assembled from the pixels of other patients, which is what `background` and `nearest_unlike` produce, is a genuine radiograph the network classifies on its own merits, and its curve stays down.

One practical warning follows. The operators whose curves recover most also tend to produce the highest degradation scores while never changing the predicted class at any step, which shows up as a sparsity rate of one. A score obtained that way rests on the gap between two curves rather than on a decision that was actually overturned, so it has to be read beside the sparsity rate and the noise onset rather than on its own.

## The search

Hill climbing with a local operator that swaps two randomly selected positions of the current ranking. A candidate is accepted when it raises the degradation score. The starting ranking is either the greedy ordering already produced by the reference sweep, at no extra cost, the sequential selection of Algorithm 3, which is stronger and quadratic in the segment count, or a random one. An explicit ranking is also accepted, which is how prior knowledge enters the procedure. The run ends after a budget of iterations or once the patience expires, and restarts from a random ranking replace an early stop when `n_restarts` is positive.

`modularity_gap` reports the largest departure from the modularity assumption of Theorem 3 along the greedy order. A value near zero says the greedy ranking is already optimal and the search has nothing left to find, and larger values say the effect of marginalizing a segment depends on what was marginalized before, which is the ordinary situation and the reason the search exists.

## Declaring the classifier

A fitted model is always required, since the explanation describes that model and nothing else. Four families are recognized without importing their libraries, so neither PyTorch nor TensorFlow becomes a dependency of the package.

| Family | Recognized through |
| --- | --- |
| Keras and TensorFlow | the call interface of the model |
| scikit-learn and any estimator following it | `predict_proba`, with the batch flattened automatically when the estimator expects one row per image |
| PyTorch | `torch.nn.Module`, evaluated under `no_grad` on the device of its parameters |
| `torchvision`, `timm`, `transformers` | the same PyTorch branch, since every image model of the three is a module |

Keras is tested before PyTorch on purpose. Under the PyTorch backend a Keras model also inherits from `torch.nn.Module`, so the reverse order would drive it through the PyTorch calling convention and bypass its own preprocessing.

Two conventions are resolved once, on a probe batch drawn from the training data, and then held fixed. The output convention states whether the raw output already lies on the probability simplex, and a softmax is applied only when it does not. The channel layout states whether the model expects the channel axis before the two spatial axes, as PyTorch does, or after them, as Keras does. Both are detected by default and both can be stated explicitly when the probe would be ambiguous, which happens when an image has as many channels as it has rows.

```python
SOFIExplainer(torch_module, X_train)                                    # detected
SOFIExplainer(keras_model, X_train)                                     # detected
SOFIExplainer(model, X_train, output="logits", channels_first=True)     # declared conventions
SOFIExplainer(model, X_train, output_fn="forward", output="logits")     # declared method
```

`output_fn` names the method of the model that produces the outputs, either as the name of an attribute or as a callable. It covers the cases where the detection would pick the wrong method or where none carries a familiar name, and the conversion of the input stays the one the framework of the model expects. A model exposing hard labels alone is rejected at construction, since probability degradation is the quantity SOFI measures. A binary network ending in one sigmoid unit is rejected for the same reason, because a single column of scores can never fall relative to an alternative, and the message says how a callable turns it into two columns.

`batch_size` bounds the number of images sent to the model in one call. A curve over a partition of many segments, multiplied by the replicas, produces batches that an image model cannot always hold, and splitting them costs nothing.

## Transformer and computer vision backbones

Every image model of `torchvision`, `timm` and `transformers` arrives through the PyTorch branch and needs no declaration. Three details separate them from a plain convolutional network, and each is handled by an argument already part of the explainer.

The first is the preprocessing. A pretrained backbone expects three channels at the resolution it was trained on, standardized with the constants of its own recipe, so the `preprocess` function is called with `channels=3` and the constants are replaced. The second is the output. A model of `transformers` returns a dataclass whose class scores sit in a `logits` field, which the wrapper unwraps on its own, and the class names are read from `config.id2label`. The third is the cost, which `batch_size` and a coarser partition keep in hand.

```python
import numpy as np
from sofiimg import SOFIExplainer

MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")   # the ImageNet recipe
STD = np.array([0.229, 0.224, 0.225], dtype="float32")
X = (preprocess(images, size=224, channels=3) - MEAN) / STD

# torchvision, any architecture, pretrained or not
import torchvision
model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
explainer = SOFIExplainer(model, X, segmentation="slic",
                          segmentation_params={"n_segments": 60, "compactness": 0.2},
                          batch_size=16)

# timm, including every vision transformer it publishes
import timm
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2).eval()
explainer = SOFIExplainer(model, X, batch_size=16, segmentation="slic",
                          segmentation_params={"n_segments": 40, "compactness": 0.2})

# transformers, whose logits and label names are read automatically
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224").eval()
explainer = SOFIExplainer(model, X, batch_size=8, segmentation="slic",
                          segmentation_params={"n_segments": 40, "compactness": 0.2})
```

```python
explainer = SOFIExplainer(
    model, X, output="logits", channels_first=True,
    output_fn=lambda batch: model(pixel_values=batch, interpolate_pos_encoding=True).logits,
)
```

The following combinations were verified end to end against this package. Pretrained weights change nothing in the plumbing, since the wrapper reads outputs alone.

| Library | Architectures verified |
| --- | --- |
| `torchvision` | ResNet-18, EfficientNet-B0, ViT-B/16, Swin-T, ConvNeXt-Tiny |
| `timm` | ViT-Tiny/16, DeiT-Tiny, ConvNeXt-Atto |
| `transformers` | `ViTForImageClassification`, `SwinForImageClassification`, `ConvNextForImageClassification` |
| Keras 3 | Sequential and functional models, under both the PyTorch and the TensorFlow backends, channels last and channels first |
| scikit-learn | `MLPClassifier` inside a pipeline that flattens the batch |

The demonstration trains one architecture twice, once in PyTorch and once in Keras over TensorFlow, and explains both. The two frameworks disagree about the channel layout and about the output convention, and the probe resolves both without a declaration, which is the point the pair is there to make.

## Preprocessing belongs outside the package

The loader returns pixels as they sit on disk. No resizing, no scaling and no normalization take place inside `sofiimg`, and this is deliberate rather than an omission.

SOFI marginalizes a segment of the array the model receives, so the array the explainer holds has to be the array the model was fitted on. Were the package to preprocess on its own, the explainer would neutralize segments of one representation while the classifier read another, and the degradation curve would measure something no one asked for. The preprocessing therefore lives in the notebook, it is applied once, and both the model and the explainer are built on its output. The same function covers a pretrained backbone by raising the channel count to three and swapping the standardization constants.

```python
def preprocess(images, size=96, channels=1):
    resized = np.stack([np.asarray(Image.fromarray(p).resize((size, size))) for p in images])
    return np.repeat((resized.astype("float32") / 255.0)[..., None], channels, axis=-1)

X_train = (preprocess(data.X_train) - CENTER) / SPREAD    # constants from the training split
```

Every figure maps whatever it is given onto the unit interval before drawing it, so a standardized array needs nothing undone by hand. When the preprocessing is not invertible and the figure should show the image a radiologist would recognize, the original image is passed through the `display` argument of `plot_segmentation`, `plot_image` and `plot_importance_map`.

## Loading data

`load_image_folder` reads the layout every image library agrees on, namely one directory per split holding one directory per class. `load_dataset` reads the bundled sample when no directory is given.

```python
load_dataset("pneumonia")                        # bundled, with its lung masks
load_dataset("my_problem", directory="~/data")   # your own files
load_image_folder("~/data", color=True)          # three channel images
load_image_folder("~/data", mask_directory="~/roi")   # with regions of interest
```

A dataset may carry a second tree of binary masks laid out identically, one file per image, which the loader returns as `masks_train` and `masks_test`. Those masks are regions of interest rather than labels, and the bundled sample ships the lung fields of every radiograph.

The images are returned in the type they were stored in, shaped `(n, height, width)` when grayscale and `(n, height, width, n_channels)` otherwise, with their labels taken from the directory names and the paths kept in `dataset.files`. Images of unequal size are refused rather than silently resized, since the resizing is a preprocessing decision.

## Figures

Every figure is reached from the object that owns it. An explanation draws itself, an experiment draws its comparisons, and no state is shared between figures. Typography, colors and sizes are arguments of each call, so a figure that needs to differ says so in its own arguments and nothing persists to the next one.

| Call | What it draws |
| --- | --- |
| `explanation.plot_explanation()` | The segments that carry the decision beside the two curves |
| `explanation.plot_segmentation()` | The image with the borders of its segments |
| `explanation.plot_marginalization()` | The image along the cumulative marginalization of the ranking |
| `explanation.plot_animation()` | The same process animated, shown in the cell and written to a file |
| `explanation.plot_overlay()` | The segments alone, without the curves |
| `explanation.plot()` | The two curves alone |
| `explanation.plot_importance_map()` | The ranking as a graded heat map |
| `study.segmentation_grid()` | One image under several partitions |

Two conventions govern every figure that draws an image.

The first concerns resolution. A classifier receives a small square, cropped and downscaled, because that is what it was trained on. A reader does not, and a partition drawn over a downscaled image hides the detail the explanation is about. Passing the original through `display` draws the partition at the resolution a reader can see, and the label map is carried onto it by nearest neighbour sampling, so no border moves and no segment appears or disappears. An `Experiment` built with `display` supplies it to every figure automatically.

The second concerns what an overlay means. Marginalizing a segment replaces its content, and drawing the replacement would show the mechanism rather than the finding. The segments are therefore painted over the untouched image and outlined, and the color carries no magnitude. By default the segments painted are those whose cumulative marginalization changes the predicted class, which is precisely the claim a sparse explanation makes.

Nothing is fixed for a session, so every figure carries its own settings and none of them leaves a trace on the next.

| Argument | Applies to | Controls |
| --- | --- | --- |
| `figsize`, `panel_size`, `width_ratio`, `ncols` | every figure | the size of the whole figure and of each panel within it |

In a grid of image panels the width is taken as given and the height is derived from it, since a panel keeps the aspect of the image it holds and a height chosen independently would leave bands of white between the panels. Passing `figsize=(10, 5)` and `figsize=(10, 10)` to the same grid therefore produces the same figure, and the panels fill their cells at any width, any number of columns and any typography.
| `font_scale`, `style` | every figure | the size of every label, tick, title and annotation, and the background |
| `display`, `cmap` | every figure holding an image | the image drawn and its colormap |
| `color`, `alpha`, `edge_color`, `linewidth` | the overlays | the fill, its opacity, its outline and the thickness of that outline in pixels |
| `k` | the overlays | how many leading segments are painted |
| `marker`, `markersize`, `marker_edge_color`, `marker_edge_width` | the curves | the markers placed on every step |
| `linewidth`, `linestyle`, `morf_color`, `lerf_color` | the curves | the two lines |
| `fill_color`, `fill_alpha` | the curves | the area between them, which is the degradation score |
| `star_marker`, `star_scale` | the curves | the marker placed where each curve changes the predicted class |
| `xlabel`, `ylabel`, `ylim`, `yticks`, `n_xticks` | the curves | the axes |
| `legend`, `legend_loc`, `legend_ncol` | the curves | the legend |
| `annotation`, `annotate_ds`, `mark_sparsity` | the curves | what is written inside the axes |

```python
explanation.plot_explanation(
    figsize=(13.0, 4.6), width_ratio=1.9, font_scale=1.3, style="white",
    color="#8e6fb6", alpha=0.6, linewidth=2,
    marker="o", markersize=5, morf_color="#b6763b", lerf_color="#4f9e6a",
    fill_color="#d9d9d9", fill_alpha=0.4, legend_ncol=1,
)
```

## Animating an explanation

`explanation.plot_animation()` replays the process a degradation curve records. The left panel adds segments in the order a ranking prescribes and the right panel grows the matching curve one point at a time, so a viewer sees which segment produced which fall. The animation renders in the notebook cell and is written to a file when `filename` is given.

The two curves are built one after the other rather than together, because they follow different orders and no single sequence of segments can drive both. Colour carries the ranking rather than the order of play, green for the segments that carry the decision and red-orange for the rest, so the pass that removes the least relevant segments first opens in red-orange and turns green at its end. The closing frames keep the segments that carry the decision, shade the area the degradation score measures and write it.

```python
explanation.plot_animation()                                   # shown in the cell
explanation.plot_animation(filename="explanation")             # also written as a GIF
explanation.plot_animation(filename="explanation", formats=("gif", "mp4"))
```


## Cost and reproducibility

Building the objective for an image costs two sweeps of the segments, one to marginalize each segment alone, which produces the greedy ordering and the anchors, and one to follow that ordering cumulatively. Every candidate proposed afterwards requires the two curves, so an evaluation costs twice a single curve. A curve issues one batched query, split into chunks of `batch_size` when one is given, and a candidate produced by a swap at positions `i` and `j` with `i` below `j` reuses the first `i` points of the MoRF curve of the incumbent and the first `n - 1 - j` points of its LeRF curve, since the reversed ranking is affected at mirrored positions. Only the affected tails are recomputed, and the result is identical to a full evaluation. Replicas multiply every query by their count.

Segment count drives everything, so it is the first knob to turn when a run is slow. Forty segments on a 96 pixel image explain a small convolutional network in seconds, while two hundred on a base transformer at 224 pixels will not.

Every run is reproducible from `random_state`, which controls the operator draws, the swap operator and the restarts. No internal state is modified during a search, so repeated calls on the same image return the same explanation.

## Reference of `SOFIExplainer`

| Parameter | Default | Accepted values | Rationale |
| --- | --- | --- | --- |
| `model` | required | A fitted classifier of any of the four families, or any object whose output method is named through `output_fn`. | The explainer is agnostic to the model family and only queries its outputs. A model is always required, since the explanation describes it. |
| `X_train` | required | Array shaped `(n, height, width)` or `(n, height, width, n_channels)`, already preprocessed as the model expects. | Source of the marginalization values. The images explained afterwards never contribute statistics. |
| `y_train` | `None` | `None` or an array of labels. | Needed by the class directed operators alone, which draw their substitution from the classes other than the one being explained. |
| `output_fn` | `None` | `None`, the name of a method of the model, or a callable receiving the batch. | Names the method that produces the outputs, for the cases where the detection would pick the wrong one. A callable takes full responsibility for the conversion of the input. |
| `output` | `"auto"` | `"auto"`, `"proba"`, `"logits"`. | Whether the model already returns probabilities. `"auto"` lets a probe batch decide, and a softmax is applied only when the raw output leaves the probability simplex. |
| `channels_first` | `"auto"` | `"auto"`, `True`, `False`. | Whether the model expects the channel axis before the two spatial axes. `"auto"` tries both orientations on the probe. |
| `classes` | `None` | `None` or a sequence of labels. | Class labels in the order of the probability columns, used for reporting alone. Read from `classes_` or from `config.id2label` when available. |
| `batch_size` | `None` | `None` or a positive integer. | Largest number of images sent to the model in one call, which bounds the memory of a curve over many segments. |
| `segmentation` | `"slic"` | Any value of the table above, or a ready `Segmentation`. | Declares which units are ranked. |
| `segmentation_params` | `None` | Dict of arguments of the procedure. | Holds `n_segments`, `compactness`, `sigma` and `patch_size`. |
| `mask` | `None` | `None` or a boolean array shaped like one image. | Region of interest shared by every image the explainer handles. Segments are grown inside it alone and everything outside stays out of every ranking. |
| `marginalization` | `"mean"` | Any operator name, a number, a `Perturbation`, `"auto"`, or a sequence of names. | Declares what neutralizing a segment means. A number is read as a constant, and `"auto"` resolves the choice per image. |
| `surge_tolerance` | `0.5` | Float in the unit interval. | Largest recovery `"auto"` tolerates once every segment is marginalized, as a fraction of the untouched response. |
| `min_drop` | `0.5` | Float in the unit interval. | Smallest fall `"auto"` requires of a candidate, which removes the operators that never move the model. |
| `marginalization_params` | `None` | Dict of arguments of the operator. | Holds `n_replicas`, `taper` and `sigma`, among others. |
| `initialization` | `"greedy"` | `"greedy"`, `"sequential"`, `"random"`, or a sequence of positions. | `"greedy"` starts from the ordering produced by the reference sweep, which is already available. `"sequential"` runs Algorithm 3, which is stronger and quadratic. An explicit sequence lets prior knowledge enter the search. |
| `max_iterations` | `200` | Non-negative integer. | Budget of proposed swaps across all restarts. Zero evaluates the initial ranking without any search. |
| `patience` | `None` | `None` or a positive integer. | Consecutive swaps without improvement tolerated before a restart or the end of the search. `None` sets it to the whole budget. |
| `n_restarts` | `0` | Integer of at least zero. | Restarts from a random ranking granted once the patience expires. |
| `accept_equal` | `False` | `False`, `True`. | Whether swaps that leave the score unchanged are accepted, which lets the search drift along plateaus. |
| `check_modularity` | `True` | `True`, `False`. | Whether the departure from the modularity assumption is measured, which costs one extra curve. |
| `random_state` | `None` | `None`, an integer, or a `numpy` `Generator`. | Seed of the operator draws, of the swap operator and of the restarts. |
| `verbose` | `True` | `True`, `False`. | Whether the one line notice naming the resolved backend, output convention and channel layout is printed when the explainer is built. |
| `progress` | `False` | `True`, `False`. | Whether a progress bar follows the search. Off by default, since a local explanation is quick and a bar per call would bury the output of a notebook. |

## Reference of `explain`

| Parameter | Default | Accepted values | Rationale |
| --- | --- | --- | --- |
| `x` | required | Array shaped `(height, width)`, `(height, width, n_channels)` or `(1, height, width, n_channels)`. | The image to explain. Segmentation applies to it alone, so a ranking describes one decision. |
| `segmentation` | `None` | `None` or a ready `Segmentation`. | Partition to rank. It is computed from the settings of the explainer when omitted, which is the usual case. A shared partition is how several images become comparable. |
| `mask` | `None` | `None` or a boolean array shaped like the image. | Region of interest for this call alone, which is the usual route when the segment differs from image to image, as an anatomical mask does. |
| `marginalization` | `None` | Any value accepted by the constructor. | Operator for this call alone. |
| `label` | `None` | `None` or a class position. | Class whose probability the search tracks. The predicted class is used by default, which makes the explanation independent of the ground truth. |
| `initialization` | `None` | `"greedy"`, `"sequential"`, `"random"`, or a sequence of positions. | Starting ranking for this call alone. |
| `max_iterations` | `None` | `None` or a non-negative integer. | Budget of proposed swaps for this call alone. Zero evaluates the initial ranking without searching. |
| `patience` | `None` | `None` or a positive integer. | Swaps without improvement tolerated for this call alone. |
| `n_restarts` | `None` | `None` or an integer of at least zero. | Restarts granted for this call alone. |
| `random_state` | `None` | `None`, an integer, or a `numpy` `Generator`. | Seed for this call alone. |
| `verbose` | `None` | `None`, `True`, `False`. | Whether a progress bar follows this search. The setting of the explainer applies when it is left out. |

`explain_batch` runs one search per image and `score_ranking` evaluates a ranking without searching.

## Reference of `SOFIExplanation`

| Attribute | Meaning |
| --- | --- |
| `ranking`, `order`, `segments` | Segments from the most to the least important, by name, by position and as objects |
| `top(k)` | First `k` segments of the ranking |
| `morf_scores`, `lerf_scores` | Reported curves of the ranking and of its reverse, inside the unit interval |
| `morf_scores_raw`, `lerf_scores_raw` | The same curves on the raw response of the model |
| `ds` | Degradation score, namely the area between the reported curves |
| `ds_raw` | The integral on the raw scale, which is the quantity the search maximizes |
| `anchor_high`, `anchor_low` | Responses placed at one and at zero when a curve is reported |
| `raw_baseline` | Probability of the explained class before any perturbation |
| `scores` | Alias of `morf_scores` |
| `auc_morf`, `auc_lerf` | Mean height of each reported curve |
| `drops` | Degradation attributable to each step of the MoRF curve |
| `sparsity_point`, `sparsity_rate`, `sparsity_probability` | Where the MoRF order changes the predicted class, and how confident the model was there |
| `lerf_sparsity_point`, `lerf_sparsity_rate` | Where the reversed order changes it, which should be far later |
| `importances` | Rank based importance in the unit interval, with the noise region excluded |
| `noise_onset`, `noise_features` | Start of the segment where fidelity recovers, and the segments it holds |
| `interpretation`, `class_directed` | What this ranking means, given the operator that produced it |
| `modularity_gap` | Departure from the assumption of Theorem 3 |
| `instance`, `states` | The image, and the image along the cumulative marginalization, always carried |
| `statistics` | Every figure of the run gathered in one dictionary |
| `summary(top_k=None)` | Report meant to be printed, listing the whole ranking, also returned by `print(explanation)` |
| `to_frame()` | Tabular view of the ranking, one row per step, with the area and centroid of each segment |
| `plot_explanation()` | The segments that carry the decision beside the two curves |
| `plot_segmentation()`, `plot_marginalization()`, `plot_animation()` | The partition, the cumulative removal and the same process animated |
| `plot()`, `plot_overlay()`, `plot_importance_map()` | The curves alone, the segments alone, and the graded heat map |
| `history` | One record per iteration of the search |
| `n_iterations`, `n_restarts_used`, `n_evaluations`, `n_model_calls` | Counters describing the run |

## Reference of `Experiment`

An experiment holds the explainer, the images, their regions of interest and the originals used for drawing, so that a comparison reads as one call.

| Parameter | Default | Accepted values | Rationale |
| --- | --- | --- | --- |
| `explainer` | required | A `SOFIExplainer`. | Its settings are the baseline each study departs from, and a study overrides one of them at a time. |
| `random_state` | `None` | `None`, an integer or a `Generator`. | Seed of the random ranking used as the lower reference of a comparison. |

Every study takes the image itself, together with an optional `mask` and `display`, exactly as `explain` does.

Each study separates the setting it varies from the way its figures are drawn. Whatever the study holds fixed is named explicitly, such as `compactness` and `sigma` in `segment_counts`, `n_segments` and `sigma` in `compactness`, and `marginalization_params` in `operators`. Everything else passes to the figures, so `figsize`, `width_ratio`, `font_scale`, `alpha`, `markersize` and the colours reach every panel the study draws.

```python
study.segment_counts(image, [10, 40, 80], compactness=0.2,      # the setting held fixed
                     figsize=(8, 4), font_scale=0.9)            # how each figure is drawn
```

| Method | What it does |
| --- | --- |
| `reliable_instances(X, y)` | Positions of the images the model classifies correctly, since the premise of the method only holds for those |
| `confident_instances(X, y, label, k)` | The same positions ordered from the most confident, optionally within one class |
| `batch(X, y, masks)` | Explains a collection and reports one row per image |
| `segment_counts(x, counts)` | Varies how many segments the image is divided into, reporting the share of the image the explanation names |
| `compactness(x, values)` | Varies how closely the segments follow the content |
| `operators(x, names)` | Varies what neutralizing a segment means, reporting the family of each operator |
| `rankings(x, saliency, orders)` | Compares SOFI against occlusion, a random order and any ranking or saliency map supplied |
| `segmentation_grid(x, counts)` | Draws the image under several partitions as one panel |

Each study returns a table with the degradation score, the fall of the response, its recovery, the sparsity rate, the step at which the class changes and the cost in queries and seconds. The explanations themselves are kept in `study.explanations_`.

## Relation to `sofits`

The two packages are deliberately parallel. Module layout, parameter names, the explainer interface, the explanation object, the metrics, the baselines and the curve figures are shared, so code written against one reads the other. Four differences follow from the data type rather than from a change of design.

| `sofits` | `sofiimg` |
| --- | --- |
| Instances shaped `(n, n_channels, n_timestamps)` | Images shaped `(n, height, width, n_channels)` |
| A segment is a channel and an interval, with `start`, `end` and `channel` | A segment is a set of pixels spanning every channel, with `area`, `bbox` and `centroid` |
| Change point detection, uniform binning, manual breakpoints | SLIC, or an arbitrary label map |
| Every stretch of the series is part of the signal | A region of interest confines the ranking, through `mask` |
| Fourteen operators, including `"linear"` | Fifteen operators, with `"blur"` and `"inpaint"` in its place |

## Data

The bundled sample under `sofiimg/datasets/pneumonia` holds 3,166 images from the paediatric chest X-ray collection of Guangzhou Women and Children's Medical Center, drawn from a public mirror released under a CC0 waiver and converted to 224 pixel grayscale JPEG, balanced between normal and pneumonia and split 2,532 for training and 634 for testing. The lung masks under `sofiimg/datasets/pneumonia_masks` are the human-drawn polygons that v7 Labs released for the same images as part of its COVID-19 X-ray dataset, rasterized to the same resolution. Both are included so that the demonstration runs offline and neither carries a clinical warranty of any kind.

## Citation

```bibtex
@article{grau2026sofits,
  title   = {Sparseness-Optimized Feature Importance for Time Series Classification},
  author  = {Grau, Isel and N{\'a}poles, Gonzalo and Jastrzebska, Agnieszka and Salgueiro, Yamisleydi},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {29874--29893},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3667092}
}

@inproceedings{grau2024sofi,
  title     = {Sparseness-Optimized Feature Importance},
  author    = {Grau, Isel and N{\'a}poles, Gonzalo},
  booktitle = {Explainable Artificial Intelligence. xAI 2024},
  series    = {Communications in Computer and Information Science},
  volume    = {2154},
  pages     = {393--415},
  publisher = {Springer},
  year      = {2024},
  doi       = {10.1007/978-3-031-63797-1_20}
}
```

## License

MIT. See `LICENSE`.

