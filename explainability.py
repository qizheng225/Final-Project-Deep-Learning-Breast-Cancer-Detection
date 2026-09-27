#TWO COMPLEMENTARY POST-HOC EXPLANATION METHODS FOR THE TRAINED CLASSIFIER:
#1. GRAD-CAM, MAPPING USING THE LAST CONVOLUTIONAL LAYER OF THE BACKBONE. USING ONLY TENSORFLOW DEPENDENCY.
#2. DEEPLIFT, IMPLEMENTED VIA SHAP'S DeepExplainer. 
#   THE STANDARD TF2-COMPATIBLE IMPLEMENTATION OF THE DEEPLIFT ALGORITHM (THE ORIGINAL `deeplift` PYPI
#   PACKAGE IS UNMAINTAINED AND DOES NOT SUPPORT TF2/KERAS3 MODELS). 
#   IF SHAP ISN'T INSTALLED, A DOCUMENTED "GRADIENT X INPUT" FALLBACK IS USED INSTEAD
#   (IT IS NOT DEEPLIFT AND IS CLEARLY LABELLED AS AN APPROXIMATION)

#BOTH RETURN A PER PIXEL ATTRIBUTION MAP THE SAME SIZE AS THE INPUT IMAGE, SO THEY CAN BE OVERLAID AND COMPARED DIRECTLY

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


#GRAD-CAM
def find_last_conv_layer(model: tf.keras.Model, base_model: Optional[tf.keras.Model] = None) -> str:
    #auto detects the name of the last 4D-output (conv-like) layer in base_model
    #uses layer.output.shape rather than the older layer.output_shape attribute, which is unreliable/unavailable for some layers under keras 3
    target = base_model if base_model is not None else model
    for layer in reversed(target.layers):
        try:
            shape = layer.output.shape
        except (AttributeError, ValueError):
            continue
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("Could not find a convolutional layer to use for Grad-CAM.")


def grad_cam(
    model: tf.keras.Model,
    image: np.ndarray,
    base_model: Optional[tf.keras.Model] = None,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    #compute a GRAD_CAM heatmap for a single image (shape [H, W, 3], unbatched, not preprocessed)
    base_model = base_model if base_model is not None else getattr(model, "base_model", None)
    if base_model is None:
        raise ValueError("Pass base_model explicitly, or use a model built with models.build_model().")

    head_layers = getattr(model, "head_layers", None)
    preprocess_fn = getattr(model, "preprocess_input", None)
    if head_layers is None or preprocess_fn is None:
        raise ValueError(
            "model is missing `.head_layers` / `.preprocess_input`.  build with "
            "models.build_model() so Grad-CAM can replay the classification head."
        )

    layer_name = last_conv_layer_name or find_last_conv_layer(model, base_model)
    conv_layer = base_model.get_layer(layer_name)

    #build a standalone model from base_model's own input/output
    last_conv_layer_model = tf.keras.Model(base_model.input, conv_layer.output)

    img_batch = tf.expand_dims(tf.convert_to_tensor(image, dtype=tf.float32), axis=0)
    preprocessed = preprocess_fn(img_batch)

    with tf.GradientTape() as tape:
        conv_outputs = last_conv_layer_model(preprocessed, training=False)
        tape.watch(conv_outputs)
        x = conv_outputs
        for layer in head_layers:
            x = layer(x, training=False)
        loss = x[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)  #ReLU
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / max_val if max_val > 0 else heatmap

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], (image.shape[0], image.shape[1])
    )
    return heatmap.numpy().squeeze() #returns a heatmap normalised to [0, 1] with shape [H, W]


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    #blend a [0,1] scaled heatmap over the original (0-255) image for display
    import matplotlib

    image = np.asarray(image)
    heatmap = np.asarray(heatmap)
    #heatmap must be 2D(H, W)
    #a common bug is an atttribute method accidentally leaving the colour-channel axis in place, which fails later
    if heatmap.ndim != 2:
        raise ValueError(
            f"overlay_heatmap expects a 2D (H, W) heatmap, got shape {heatmap.shape}. "
            "This usually means an attribution map still has a colour-channel axis - "
            "check the axis being summed/collapsed before calling overlay_heatmap()."
        )

    if image.max() > 1.0:
        image_norm = image / 255.0
    else:
        image_norm = image

    try:
        cmap = matplotlib.colormaps["jet"]
    except AttributeError:
        import matplotlib.cm as cm

        cmap = cm.get_cmap("jet")

    colored_heatmap = cmap(heatmap)[:, :, :3]
    overlaid = colored_heatmap * alpha + image_norm * (1 - alpha)
    return np.clip(overlaid, 0, 1)


#DEEPLIFT (via SHAP DeepExplainer, with a documented fallback)
def _model_has_shap_incompatible_ops(model: tf.keras.Model) -> bool:
    #shap.DeepExplainer lacks a custom gradient for DepthwiseConv2dNative, used by EfficientNet's MBConv blocks
    #attempting DeepExplainer on those models can corrupt TensorFlow's global gradient registry, breaking subsequent GradientTape calls
    #hence, this detect incompatibility before constructing DeepExplainer, since try/except cannot safely recover afterward
    base_model = getattr(model, "base_model", model)
    return any(isinstance(layer, tf.keras.layers.DepthwiseConv2D) for layer in base_model.layers)


#annotating figures with the label of the method that was used
DEEPLIFT_LABEL = "DeepLIFT"
DEEPLIFT_FALLBACK_LABEL = "DeepLIFT (approx. - grad x input)"


def deeplift_attributions(
    model: tf.keras.Model,
    images: np.ndarray,
    background: np.ndarray,
    return_method: bool = False,
) -> np.ndarray:
    #per pixel DeepLIFT attributions relative to background, summed over colour channels for direct comparison with Grad-CAM.
    #if return_method=True, also returns the method label. default preserves existing callers/tests
    def _done(result, label):
        return (result, label) if return_method else result

    if _model_has_shap_incompatible_ops(model):
        print(
            "[explainability] Skipping shap.DeepExplainer: this backbone uses "
            "DepthwiseConv2D layers (e.g. EfficientNet's MBConv blocks), which "
            "SHAP's TF DeepExplainer has no gradient rule for and which is known "
            "to corrupt TensorFlow's gradient registry for the rest of the "
            "process if attempted. Using the gradient x input fallback directly."
        )
        return _done(_gradient_x_input_fallback(model, images, background), DEEPLIFT_FALLBACK_LABEL)

    try:
        import shap

        explainer = shap.DeepExplainer(model, background)
        #disable SHAP's additivity check, which can false-positive on TF2/Keras3 graphs with preprocessing, augmentation, or nested submodels
        try:
            shap_values = explainer.shap_values(images, check_additivity=False)
        except TypeError:
            #older shap versions dont accept the check_additivity KWARG
            shap_values = explainer.shap_values(images)
        #handle both older list output and newer arrays with a trailing num_outputs axis before collapsing the colour channel
        attributions = shap_values[0] if isinstance(shap_values, list) else shap_values
        attributions = np.asarray(attributions)
        if attributions.ndim == np.asarray(images).ndim + 1:
            attributions = attributions[..., 0]  # drop the trailing num_outputs axis
        return _done(np.sum(attributions, axis=-1), DEEPLIFT_LABEL)  # collapse RGB channels
    except ImportError:
        return _done(_gradient_x_input_fallback(model, images, background), DEEPLIFT_FALLBACK_LABEL)
    except Exception as exc:
         #fall back on SHAP graph/compatibility errors rather than failing the entire explainability pipeline.
        print(f"[explainability] shap.DeepExplainer failed ({exc!r}); using gradient x input fallback.")
        return _done(_gradient_x_input_fallback(model, images, background), DEEPLIFT_FALLBACK_LABEL)


def _gradient_x_input_fallback(
    model: tf.keras.Model, images: np.ndarray, background: np.ndarray
) -> np.ndarray:
    #plain gradient x input saliency map, not true deeplift
    baseline = tf.reduce_mean(tf.convert_to_tensor(background, dtype=tf.float32), axis=0)
    img_tensor = tf.convert_to_tensor(images, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor, training=False)[:, 0]

    grads = tape.gradient(preds, img_tensor)
    attributions = grads * (img_tensor - baseline)
    return tf.reduce_sum(attributions, axis=-1).numpy()


def normalize_attribution_map(attr_map: np.ndarray) -> np.ndarray:
    #rescale a (possibly signed) attribution map to [0, 1] for visual comparison
    attr_map = np.asarray(attr_map, dtype=np.float32)
    attr_map = np.abs(attr_map)
    max_val = attr_map.max()
    return attr_map / max_val if max_val > 0 else attr_map


#SIDE-BY-SIDE COMPARISON
def compare_gradcam_and_deeplift(
    model: tf.keras.Model,
    image: np.ndarray,
    background: np.ndarray,
    base_model: Optional[tf.keras.Model] = None,
    return_method: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    #compares Grad-CAM and DeepLIFT, returning both heatmaps normalised to [0, 1]
    #if return_method=True, also returns the DeepLIFT method used
    cam = grad_cam(model, image, base_model=base_model)

    deeplift_raw, deeplift_method = deeplift_attributions(
        model, np.expand_dims(image, 0), background, return_method=True
    )
    deeplift_map = normalize_attribution_map(deeplift_raw[0])

    if return_method:
        return cam, deeplift_map, deeplift_method
    return cam, deeplift_map


def plot_gradcam_vs_deeplift(image, gradcam_map, deeplift_map, ax=None, deeplift_label: str = DEEPLIFT_LABEL):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    img_norm = image / 255.0 if np.asarray(image).max() > 1.0 else image
    ax[0].imshow(img_norm)
    ax[0].set_title("Original")
    ax[1].imshow(overlay_heatmap(image, gradcam_map))
    ax[1].set_title("Grad-CAM")
    ax[2].imshow(overlay_heatmap(image, deeplift_map))
    #uses the actual DeepLIFT method label to avoid mislabelling fallback attributions
    ax[2].set_title(deeplift_label)
    for a in ax:
        a.axis("off")
    return ax


#CREATES A MULTI-ROW EXPLAINABILITY GRID SHOWING Original | Grad-CAM | DeepLIFT FOR DIFFERENT PREDICTION OUTCOMES
def plot_explainability_grid(images, gradcam_maps, deeplift_maps, row_labels, save_path=None, deeplift_label: str = DEEPLIFT_LABEL):
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    #uses the actual attribution method label so the third column is correctly identified as DeepLIFT or the fallback method
    col_titles = ["Original", "Grad-CAM", deeplift_label]
    for i in range(n):
        img_norm = images[i] / 255.0 if np.asarray(images[i]).max() > 1.0 else images[i]
        panels = [img_norm, overlay_heatmap(images[i], gradcam_maps[i]), overlay_heatmap(images[i], deeplift_maps[i])]
        for j, panel in enumerate(panels):
            axes[i, j].imshow(panel)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(col_titles[j])
        axes[i, 0].set_ylabel(row_labels[i], fontsize=11)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig