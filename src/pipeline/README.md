## Usage of this package via orchestrator.py

### Synthetic Few-Shot (no segmentation) 
```
results = orchestrate(
    mode="few_shot",
    generate_images_fn=generate_images,
    labels=["car", "phone", "person", "computer", "bag", "jacket"],
    n_per_label_train=20,
    n_per_label_test=6,
    store_root=DEFAULT_OUTPUT_ROOT,
    classifier_path=DEFAULT_CLASSIFIER_PATH,
    zero_shot_kwargs={"sam": None},
    verbose=True
)
print("-------------- Synthetic Few Shot Results: -----------------------\n")
print(results)

```  
#### Synthetic Few Shot with an existing classifier
```
results = orchestrate(
    mode="few_shot",
    generate_images_fn=generate_images,
    labels=["car", "phone", "person", "computer", "bag", "jacket"],
    n_per_label_train=20,
    n_per_label_test=6,
    store_root=DEFAULT_OUTPUT_ROOT,
    classifier_path=DEFAULT_CLASSIFIER_PATH,
    use_existing_classifier=True,
    zero_shot_kwargs={"sam": None},
    verbose=True
)
print("-------------- Synthetic Few Shot Results: -----------------------\n")
print(results)

```
  
### Synthetic Few-Shot Test Results:  
```
{'summary': {'train_summary': {'car': {'generated_images': 20, 'stored_samples': 20, 'store_root': '/home/passwd/AI_object_detection_app/src/pipeline/store/data', 'elapsed_time': 263.51286339759827}, 'phone': {'generated_images': 20, 'stored_samples': 20, 'store_root': '/home/passwd/AI_object_detection_app/src/pipeline/store/data', 'elapsed_time': 262.9686484336853}, 'person': {'generated_images': 20, 'stored_samples': 20, 'store_root': '/home/passwd/AI_object_detection_app/src/pipeline/store/data', 'elapsed_time': 245.06652808189392}, 'computer': {'generated_images': 20, 'stored_samples': 20, 'store_root': '/home/passwd/AI_object_detection_app/src/pipeline/store/data', 'elapsed_time': 257.27052998542786}, 'bag': {'generated_images': 20, 'stored_samples': 20, 'store_root': '/home/passwd/AI_object_detection_app/src/pipeline/store/data', 'elapsed_time': 260.0440080165863}, 'jacket': {'generated_images': 20, 'stored_samples': 20, 'store_root': '/home/passwd/AI_object_detection_app/src/pipeline/store/data', 'elapsed_time': 263.44989109039307}}, 'prototypes': {'jacket': {'label': 'jacket', 'prototype_path': 'src/pipeline/store/data/prototypes/jacket/prototype.npy', 'count': 30, 'feature_dim': 3072, 'created_at': 1757865945.1890006}, 'bag': {'label': 'bag', 'prototype_path': 'src/pipeline/store/data/prototypes/bag/prototype.npy', 'count': 30, 'feature_dim': 3072, 'created_at': 1757865945.2022674}, 'phone': {'label': 'phone', 'prototype_path': 'src/pipeline/store/data/prototypes/phone/prototype.npy', 'count': 20, 'feature_dim': 3072, 'created_at': 1757865945.20863}, 'cat': {'label': 'cat', 'prototype_path': 'src/pipeline/store/data/prototypes/cat/prototype.npy', 'count': 10, 'feature_dim': 3072, 'created_at': 1757865945.2142062}, 'computer': {'label': 'computer', 'prototype_path': 'src/pipeline/store/data/prototypes/computer/prototype.npy', 'count': 30, 'feature_dim': 3072, 'created_at': 1757865945.2273817}, 'car': {'label': 'car', 'prototype_path': 'src/pipeline/store/data/prototypes/car/prototype.npy', 'count': 20, 'feature_dim': 3072, 'created_at': 1757865945.23489}, 'person': {'label': 'person', 'prototype_path': 'src/pipeline/store/data/prototypes/person/prototype.npy', 'count': 20, 'feature_dim': 3072, 'created_at': 1757865945.2395709}}, 'classifier_info': {'num_samples': 160, 'classes': [np.str_('bag'), np.str_('car'), np.str_('cat'), np.str_('computer'), np.str_('jacket'), np.str_('person'), np.str_('phone')], 'classifier_path': '/home/passwd/AI_object_detection_app/src/pipeline/store/classifier/fewshot_clf.joblib'}, 'metrics': {'overall': {'accuracy': 0.8055555555555556, 'precision': 0.8678571428571429, 'recall': 0.8055555555555555, 'f1': 0.782814407814408}, 'per_label': {'bag': {'precision': 0.8571428571428571, 'recall': 1.0, 'f1': 0.9230769230769231, 'support': 6}, 'car': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 6}, 'computer': {'precision': 0.6, 'recall': 1.0, 'f1': 0.75, 'support': 6}, 'jacket': {'precision': 0.75, 'recall': 1.0, 'f1': 0.8571428571428571, 'support': 6}, 'person': {'precision': 1.0, 'recall': 0.3333333333333333, 'f1': 0.5, 'support': 6}, 'phone': {'precision': 1.0, 'recall': 0.5, 'f1': 0.6666666666666666, 'support': 6}, 'macro avg': {'precision': 0.8678571428571429, 'recall': 0.8055555555555555, 'f1': 0.782814407814408, 'support': 36}, 'weighted avg': {'precision': 0.8678571428571428, 'recall': 0.8055555555555556, 'f1': 0.7828144078144078, 'support': 36}}}, 'num_test_samples': 36, 'avg_inference_time': 0.0009715954462687174, 'total_elapsed_time': 1976.5016734600067}, 'per_sample_results': [{'test_image': 'src/synthimage/generated_images/car/car_0001_1.png', 'true_label': 'car', 'pred_label': np.str_('car'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.33618879318237305}}, {'test_image': 'src/synthimage/generated_images/car/car_0002_1.png', 'true_label': 'car', 'pred_label': np.str_('car'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.3628697395324707}}, {'test_image': 'src/synthimage/generated_images/car/car_0003_1.png', 'true_label': 'car', 'pred_label': np.str_('car'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.35098862648010254}}, {'test_image': 'src/synthimage/generated_images/car/car_0004_1.png', 'true_label': 'car', 'pred_label': np.str_('car'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.30631351470947266}}, {'test_image': 'src/synthimage/generated_images/car/car_0005_1.png', 'true_label': 'car', 'pred_label': np.str_('car'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.37966036796569824}}, {'test_image': 'src/synthimage/generated_images/car/car_0006_1.png', 'true_label': 'car', 'pred_label': np.str_('car'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.32896876335144043}}, {'test_image': 'src/synthimage/generated_images/phone/phone_0001_1.png', 'true_label': 'phone', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.3895559310913086}}, {'test_image': 'src/synthimage/generated_images/phone/phone_0002_1.png', 'true_label': 'phone', 'pred_label': np.str_('phone'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.34172534942626953}}, {'test_image': 'src/synthimage/generated_images/phone/phone_0003_1.png', 'true_label': 'phone', 'pred_label': np.str_('phone'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.33501219749450684}}, {'test_image': 'src/synthimage/generated_images/phone/phone_0004_1.png', 'true_label': 'phone', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.47122979164123535}}, {'test_image': 'src/synthimage/generated_images/phone/phone_0005_1.png', 'true_label': 'phone', 'pred_label': np.str_('phone'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.3565824031829834}}, {'test_image': 'src/synthimage/generated_images/phone/phone_0006_1.png', 'true_label': 'phone', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.41861677169799805}}, {'test_image': 'src/synthimage/generated_images/person/person_0001_1.png', 'true_label': 'person', 'pred_label': np.str_('person'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.3268580436706543}}, {'test_image': 'src/synthimage/generated_images/person/person_0002_1.png', 'true_label': 'person', 'pred_label': np.str_('jacket'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.33762073516845703}}, {'test_image': 'src/synthimage/generated_images/person/person_0003_1.png', 'true_label': 'person', 'pred_label': np.str_('bag'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.3958559036254883}}, {'test_image': 'src/synthimage/generated_images/person/person_0004_1.png', 'true_label': 'person', 'pred_label': np.str_('jacket'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.499645471572876}}, {'test_image': 'src/synthimage/generated_images/person/person_0005_1.png', 'true_label': 'person', 'pred_label': np.str_('person'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.44434022903442383}}, {'test_image': 'src/synthimage/generated_images/person/person_0006_1.png', 'true_label': 'person', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.31955862045288086}}, {'test_image': 'src/synthimage/generated_images/computer/computer_0001_1.png', 'true_label': 'computer', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.357668399810791}}, {'test_image': 'src/synthimage/generated_images/computer/computer_0002_1.png', 'true_label': 'computer', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.40389251708984375}}, {'test_image': 'src/synthimage/generated_images/computer/computer_0003_1.png', 'true_label': 'computer', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.43491625785827637}}, {'test_image': 'src/synthimage/generated_images/computer/computer_0004_1.png', 'true_label': 'computer', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.6018550395965576}}, {'test_image': 'src/synthimage/generated_images/computer/computer_0005_1.png', 'true_label': 'computer', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.38329076766967773}}, {'test_image': 'src/synthimage/generated_images/computer/computer_0006_1.png', 'true_label': 'computer', 'pred_label': np.str_('computer'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.43892931938171387}}, {'test_image': 'src/synthimage/generated_images/bag/bag_0001_1.png', 'true_label': 'bag', 'pred_label': np.str_('bag'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.37025022506713867}}, {'test_image': 'src/synthimage/generated_images/bag/bag_0002_1.png', 'true_label': 'bag', 'pred_label': np.str_('bag'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.466418981552124}}, {'test_image': 'src/synthimage/generated_images/bag/bag_0003_1.png', 'true_label': 'bag', 'pred_label': np.str_('bag'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.414858341217041}}, {'test_image': 'src/synthimage/generated_images/bag/bag_0004_1.png', 'true_label': 'bag', 'pred_label': np.str_('bag'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.565911054611206}}, {'test_image': 'src/synthimage/generated_images/bag/bag_0005_1.png', 'true_label': 'bag', 'pred_label': np.str_('bag'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.40828943252563477}}, {'test_image': 'src/synthimage/generated_images/bag/bag_0006_1.png', 'true_label': 'bag', 'pred_label': np.str_('bag'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.375103235244751}}, {'test_image': 'src/synthimage/generated_images/jacket/jacket_0001_1.png', 'true_label': 'jacket', 'pred_label': np.str_('jacket'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.33396315574645996}}, {'test_image': 'src/synthimage/generated_images/jacket/jacket_0002_1.png', 'true_label': 'jacket', 'pred_label': np.str_('jacket'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.32880091667175293}}, {'test_image': 'src/synthimage/generated_images/jacket/jacket_0003_1.png', 'true_label': 'jacket', 'pred_label': np.str_('jacket'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.37245774269104004}}, {'test_image': 'src/synthimage/generated_images/jacket/jacket_0004_1.png', 'true_label': 'jacket', 'pred_label': np.str_('jacket'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.4021151065826416}}, {'test_image': 'src/synthimage/generated_images/jacket/jacket_0005_1.png', 'true_label': 'jacket', 'pred_label': np.str_('jacket'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.37629151344299316}}, {'test_image': 'src/synthimage/generated_images/jacket/jacket_0006_1.png', 'true_label': 'jacket', 'pred_label': np.str_('jacket'), 'feat_meta': {'model_name': 'resnet50', 'device': 'cpu', 'feature_dim': 3072, 'tta': True, 'tta_augments': 2, 'inference_time': 0.33884263038635254}}]}

```  
  
### Synthetic Zero-Shot (no segmentation)  
```
results = orchestrate(
    mode="zero_shot",
    generate_images_fn=generate_images,
    labels=["car", "phone"],
    n_per_label_test=6,
    candidate_labels=["car", "phone", "person"],
    use_prototypes=False,
    use_clip=True,
    zero_shot_kwargs={"sam": None, "clip_model_name": CLIP_MODEL, "device": DEVICE},
    verbose=True
)
print("-------------- Zero Shot Results: -----------------------\n")
print(results["metrics"])


```  

### Synthetic Zero-Shot Test Results:  
```
-------------- Synthetic Zero Shot Results: -----------------------

{'clip': {'overall': {'accuracy': 0.8333333333333334, 'precision': 0.6111111111111112, 'recall': 0.5555555555555556, 'f1': 0.5808080808080808}, 'per_label': {'car': {'precision': 1.0, 'recall': 0.8333333333333334, 'f1': 0.9090909090909091, 'support': 6}, 'person': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}, 'phone': {'precision': 0.8333333333333334, 'recall': 0.8333333333333334, 'f1': 0.8333333333333334, 'support': 6}, 'macro avg': {'precision': 0.6111111111111112, 'recall': 0.5555555555555556, 'f1': 0.5808080808080808, 'support': 12}, 'weighted avg': {'precision': 0.9166666666666666, 'recall': 0.8333333333333334, 'f1': 0.8712121212121211, 'support': 12}}}}
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
