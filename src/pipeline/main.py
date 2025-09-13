"""
Example entrypoint demonstrating use of the modular pipeline.

This script:
- Loads an image
- Runs SAM to get masks + metadata
- Demonstrates ResNet feature extraction, prototype building, and prototype classification
- Demonstrates CLIP zero-shot classification (if CLIP installed)
- Demonstrates few-shot training using extracted features (LogisticRegression)
"""
from PIL import Image
import numpy as np

from .models.sam_model import SAMWrapper
from .models.resnet_model import ResNetWrapper
from .models.zero_shot import CLIPZeroShot
from .models.few_shot import FewShotClassifier
from .utils.plotting import show_image, show_masks_on_image

def run_example(image_path: str = "image.png"):
    img = Image.open(image_path).convert("RGB")

    # 1) SAM inference
    sam = SAMWrapper()
    masks, sam_meta = sam.infer(img)
    print("SAM metadata:", sam_meta)
    show_masks_on_image(img, masks)

    # 2) ResNet features
    resnet = ResNetWrapper()
    features, res_meta = resnet.extract_features(img)
    print("ResNet extract metadata:", res_meta)
    print("Feature length:", features.shape)

    # 3) Prototype demo: assume we have sample images per label (here we reuse the same image as a toy example)
    resnet.add_prototype(img, label="demo_label")
    resnet.finalize_prototypes()
    proto_result, proto_meta = resnet.classify_with_prototypes(img)
    print("Prototype classification result:", proto_result)
    print("Prototype metadata:", proto_meta)

    # 4) CLIP zero-shot (if clip installed)
    try:
        clip_zs = CLIPZeroShot()
        candidate_labels = ["cat", "dog", "car", "person"]
        zs_result, zs_meta = clip_zs.classify(img, candidate_labels)
        print("CLIP zero-shot result:", zs_result)
        print("CLIP metadata:", zs_meta)
    except Exception as e:
        print("CLIP zero-shot unavailable (install clip to enable). Error:", e)

    # 5) Few-shot demo (toy): train a logistic regression with one positive & one negative sample
    from PIL import Image
    # Toy dataset: repeating the same image for class A and generating jitter for class B could be used.
    # In real usage: collect small labeled dataset and extract features via resnet.extract_features
    fs = FewShotClassifier()
    # For demo we extract features twice and give two different labels (not meaningful)
    feat1, _ = resnet.extract_features(img)
    feat2, _ = resnet.extract_features(img)  # in practice use different images
    X_train = [feat1, feat2]
    y_train = ["A", "B"]
    fs.fit(X_train, y_train)
    preds = fs.predict(X_train)
    metrics = fs.evaluate(y_train, preds)
    print("Few-shot training preds:", preds)
    print("Few-shot metrics:", metrics)

if __name__ == "__main__":
    run_example("image.png")
