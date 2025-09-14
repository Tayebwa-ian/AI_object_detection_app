## Usage of this package via orchestrator.py

### Synthetic Few-Shot (no segmentation) 
```
results = orchestrate(
    mode="few_shot",
    generate_images_fn=generate_images,
    labels=["car","phone"],
    n_per_label_train=5,
    n_per_label_test=10,
    store_root="feature_store",
    classifier_path="models/fewshot.joblib"
)
print(results["summary"])

```  
  
### Synthetic Zero-Shot (no segmentation)  
```
results = orchestrate(
    mode="zero_shot",
    generate_images_fn=generate_images,
    labels=["car","phone"],
    n_per_label_test=10,
    candidate_labels=["car","phone","person"],
    store_root="feature_store"
)
print(results["metrics"])

```  
  
### User Few-Shot (segmentation applied)  
```
results = orchestrate(
    mode="user_few_shot",
    image_path="user_image.jpg",
    store_root="feature_store",
    classifier_path="models/fewshot.joblib"
)
print(results["counts"])

```  
  
### User Zero-Shot (segmentation applied)  
```
results = orchestrate(
    mode="user_zero_shot",
    image_path="user_image.jpg",
    candidate_labels=["car","phone","person"],
    store_root="feature_store"
)
print(results["counts"])

```  
