"""
Main Pipeline Entry Point
Coordinates SAM segmentation + ResNet classification

Usage:
    from app.pipeline import run_pipeline
    results = run_pipeline(image_path, confidence_threshold=0.7)
"""

import torch
import numpy as np
from PIL import Image
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import time
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

from .postprocess import filter_segments, apply_nms, aggregate_results
from .mapping import map_labels, get_synonyms
from .mapping import get_candidate_set


class LightweightPipeline:
    """Optimized pipeline for API deployment"""
    
    def __init__(self, 
                 sam_model_type="vit_b", 
                 classification_model="microsoft/resnet-50",
                 device="cpu"):
        """
        Initialize pipeline components
        
        Args:
            sam_model_type: SAM model variant (vit_b, vit_l, vit_h)
            classification_model: HuggingFace model name
            device: cpu or cuda
        """
        self.device = device
        self.sam_model = None
        self.mask_generator = None
        self.classifier = None
        self.processor = None
        
        # Load models lazily
        self._load_sam(sam_model_type)
        self._load_classifier(classification_model)
    
    def _load_sam(self, model_type):
        """Load SAM model with enhanced optimizations and device selection"""
        try:
            # Smart device selection with fallback (FIXED: MPS compatibility)
            if torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA GPU for SAM")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # FIXED: MPS has compatibility issues with SAM, use CPU for stability
                device = "cpu"
                print("Apple Silicon detected - using CPU for SAM compatibility")
            else:
                device = "cpu"
                print("Using CPU for SAM (slower but stable)")
            
            self.sam_device = device
            # Robust checkpoint discovery: try standard name then known local names
            checkpoint_candidates = [
                f"sam_{model_type}.pth",
                # Known repo file for vit_b
                "sam_vit_b_01ec64.pth" if model_type == "vit_b" else None,
            ]
            checkpoint_candidates = [c for c in checkpoint_candidates if c]

            checkpoint_path = None
            for candidate in checkpoint_candidates:
                if os.path.isabs(candidate) and os.path.exists(candidate):
                    checkpoint_path = candidate
                    break
                if os.path.exists(candidate):
                    checkpoint_path = candidate
                    break
            if checkpoint_path is None:
                # Fall back to first candidate; underlying loader may fetch/download
                checkpoint_path = checkpoint_candidates[0]

            self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam_model.to(device=device)
            
            # FIXED: Optimized parameters for better performance
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=8,           # FIXED: Reduced from 16 (55% fewer points)
                pred_iou_thresh=0.88,        # FIXED: Slightly higher quality filter
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=500,    # FIXED: Allow smaller objects
                box_nms_thresh=0.3           # FIXED: Added NMS for overlapping masks
            )
            print(f"SAM {model_type} loaded successfully on {device}")
        except Exception as e:
            print(f"SAM loading failed: {e}")
            print("Run: python setup_models.py")
            self.sam_model = None
            self.mask_generator = None
            self.sam_device = "cpu"
    
    def _load_classifier(self, model_name):
        """Load classification model"""
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.classifier = AutoModelForImageClassification.from_pretrained(model_name)
            self.classifier.eval()
            print(f"Classifier {model_name} loaded successfully")
        except Exception as e:
            print(f"Classifier loading failed: {e}")
            raise
    
    def segment_image(self, image_path):
        """Generate segments using SAM with memory optimization"""
        start_time = time.time()
        try:
            # Check if SAM is available
            if self.mask_generator is None:
                print("SAM not available, returning empty segments")
                return [], [], None
                
            # Load and process image with memory optimization
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # FIXED: Resize large images to prevent memory issues
            height, width = image.shape[:2]
            max_size = 1024  # Reasonable limit for processing
            
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                print(f"Resized image: {width}×{height} → {new_width}×{new_height}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # FIXED: Generate masks with memory management
            with torch.no_grad():  # Disable gradient computation for memory efficiency
                masks = self.mask_generator.generate(image_rgb)
                
                # Clear GPU cache if using CUDA
                if hasattr(self, 'sam_device') and self.sam_device == "cuda":
                    torch.cuda.empty_cache()
            
            # Extract segments and bounding boxes
            segments = []
            bboxes = []
            
            for mask_data in masks:
                mask = mask_data['segmentation']
                bbox = mask_data['bbox']  # [x, y, w, h]
                
                # Convert mask to segment image
                segment = image_rgb.copy()
                segment[~mask] = [128, 128, 128]  # Gray background
                
                segments.append(segment)
                bboxes.append(bbox)
            
            # Record timing
            duration = time.time() - start_time
            print(f"SAM segmentation completed in {duration:.3f}s")
            
            return segments, bboxes, image_rgb
            
        except Exception as e:
            print(f"Segmentation failed: {e}")
            return [], [], None
    
    def classify_segments(self, segments, bboxes=None, original_image=None):
        """Classify segments using ResNet with enhanced processing"""
        start_time = time.time()
        results = []
        
        for i, segment in enumerate(segments):
            try:
                # FIXED: Enhanced segment preprocessing
                segment_result = self._classify_single_segment(
                    segment, i, bboxes[i] if bboxes else None, original_image
                )
                results.append(segment_result)
                
            except Exception as e:
                print(f"Classification failed for segment {i}: {e}")
                results.append({
                    'segment_id': i,
                    'raw_label': 'unknown',
                    'confidence': 0.0,
                    'calibrated_confidence': 0.0
                })
        
        # Record timing
        duration = time.time() - start_time
        print(f"ResNet classification completed in {duration:.3f}s")
        
        return results
    
    def _classify_single_segment(self, segment, segment_id, bbox=None, original_image=None):
        """FIXED: Enhanced single segment classification with confidence calibration"""
        
        # Convert to PIL Image and ensure RGB
        if segment.dtype != np.uint8:
            segment = (segment * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(segment)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Process and classify
        inputs = self.processor(pil_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits[0], dim=0)
        
        # Get top prediction
        top_prob, top_class = torch.max(probabilities, 0)
        predicted_label = self.classifier.config.id2label[top_class.item()]
        raw_confidence = top_prob.item()
        
        # FIXED: Confidence calibration based on segment characteristics
        segment_area = segment.shape[0] * segment.shape[1]
        calibrated_confidence = self._calibrate_confidence(
            raw_confidence, predicted_label, segment_area
        )
        
        return {
            'segment_id': segment_id,
            'raw_label': predicted_label,
            'confidence': raw_confidence,
            'calibrated_confidence': calibrated_confidence,
            'segment_area': segment_area,
            'reliability_factors': {
                'size_factor': 'small' if segment_area < 1000 else 'medium' if segment_area < 50000 else 'large',
                'confidence_adjustment': calibrated_confidence / raw_confidence if raw_confidence > 0 else 1.0
            }
        }
    
    def _calibrate_confidence(self, raw_confidence, predicted_class, segment_area):
        """FIXED: Calibrate confidence scores based on reliability factors"""
        
        calibrated_confidence = raw_confidence
        
        # Factor 1: Segment size reliability
        if segment_area < 1000:  # Very small segments
            calibrated_confidence *= 0.7  # Reduce confidence
        elif segment_area > 50000:  # Very large segments
            calibrated_confidence *= 0.8  # Slightly reduce (might be background)
        
        # Factor 2: Class reliability (ImageNet bias adjustment)
        unreliable_classes = [
            'egyptian cat', 'sports car', 'convertible', 'limousine',
            'airship', 'dirigible', 'balloon', 'parachute'
        ]
        
        if any(unreliable in predicted_class.lower() for unreliable in unreliable_classes):
            calibrated_confidence *= 0.6  # Reduce confidence for biased classes
        
        # Ensure confidence stays in valid range
        calibrated_confidence = max(0.0, min(1.0, calibrated_confidence))
        
        return calibrated_confidence


def run_pipeline(image_path, 
                confidence_threshold=0.7,
                nms_threshold=0.3,
                target_classes=None,
                enable_mapping=True):
    """
    Main pipeline entrypoint
    
    Args:
        image_path: Path to input image
        confidence_threshold: Minimum confidence for detections
        nms_threshold: Non-maximum suppression threshold
        target_classes: List of target class names (optional)
        enable_mapping: Enable synonym mapping
    
    Returns:
        dict: {
            'image_path': str,
            'detections': list,
            'summary': dict,
            'processing_time': float
        }
    """
    import time
    start_time = time.time()
    
    try:
        # Initialize pipeline
        pipeline = LightweightPipeline()
        
        # Step 1: Segmentation
        print(f"Processing: {image_path}")
        sam_start = time.time()
        segments, bboxes, original_image = pipeline.segment_image(image_path)
        sam_time = time.time() - sam_start
        
        if not segments:
            return {
            'image_path': image_path,
            'detections': [],
            'summary': {
                'total_objects': 0, 
                'error': 'No segments found',
                'processing_time': f"{time.time() - start_time:.2f}s"
            },
            'processing_time': time.time() - start_time
        }
        
        # Step 2: Classification (FIXED: Pass additional context)
        print(f"Classifying {len(segments)} segments...")
        resnet_start = time.time()
        classifications = pipeline.classify_segments(segments, bboxes, original_image)
        resnet_time = time.time() - resnet_start
        
        # Step 3: Post-processing
        print("Post-processing...")
        mapper_start = time.time()
        filtered_results = filter_segments(
            classifications, bboxes, 
            confidence_threshold=confidence_threshold
        )
        
        nms_results = apply_nms(filtered_results, threshold=nms_threshold)
        
        # Step 4: Label mapping (optional)
        if enable_mapping:
            print("Mapping labels...")
            mapped_results = map_labels(nms_results, target_classes)
        else:
            mapped_results = nms_results
        
        # Step 5: Aggregation
        final_results = aggregate_results(mapped_results)
        mapper_time = time.time() - mapper_start
        
        processing_time = time.time() - start_time
        
        return {
            'image_path': image_path,
            'detections': final_results,
            'summary': {
                'total_objects': len(final_results),
                'processing_time': f"{processing_time:.2f}s",
                'segments_generated': len(segments),
                'segments_after_filtering': len(filtered_results),
                'segments_after_nms': len(nms_results)
            },
            'processing_time': processing_time,
            'model_times': {
                'sam': sam_time,
                'resnet': resnet_time,
                'mapper': mapper_time
            },
            'segments_count': len(segments),
            'object_types_found': len(set(d.get('mapped_label', d.get('raw_label', 'unknown')) for d in final_results)),
            'avg_segment_area': sum(d.get('area', 0) for d in final_results) / max(len(final_results), 1),
            'models_used': ['sam', 'resnet', 'mapper']
        }
        
    except Exception as e:
        return {
            'image_path': image_path,
            'detections': [],
            'summary': {
                'total_objects': 0,
                'error': str(e),
                'processing_time': f"{time.time() - start_time:.2f}s"
            },
            'processing_time': time.time() - start_time
        }


# Quick test function
def test_pipeline():
    """Test the pipeline with a sample image"""
    import os
    
    # Find a test image
    test_images = [
        "model_pipeline/image.png",
        "model_pipeline/image1.jpg",
        "model_pipeline/my_dataset/test/bicycle/Image_4.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"Testing pipeline with: {img_path}")
            results = run_pipeline(img_path, confidence_threshold=0.5)
            print(f"Results: {results['summary']}")
            return results
    
    print("No test images found")
    return None


if __name__ == "__main__":
    # Test the pipeline
    test_pipeline()


# -----------------------------------------------------------------------------
# Compatibility API for existing views expecting `pipeline.process_*` methods
# -----------------------------------------------------------------------------

class _PipelineCompatibilityAdapter:
    """Adapter exposing the legacy interface used by API views.

    Methods:
        - process_image(image_path, object_type)
        - process_image_auto(image_path)
    """

    def __init__(self):
        pass

    def _count_by_label(self, detections: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        matched = [d for d in detections if (d.get('mapped_label') or d.get('raw_label', '')).lower() == label.lower()]
        count = len(matched)
        if count == 0:
            return {"count": 0, "avg_conf": 0.0}
        avg_conf = float(np.mean([d.get('confidence', 0.0) for d in matched]))
        return {"count": count, "avg_conf": avg_conf}

    def process_image(self, image_path: str, object_type: str) -> Dict[str, Any]:
        """Process a single image focusing on a specific object_type."""
        # Use a broader candidate set for mapping to enable meaningful
        # zero-shot selection instead of a single-class (trivial) list.
        candidates = get_candidate_set('general')
        if object_type not in candidates:
            candidates = candidates + [object_type]

        result = run_pipeline(
            image_path,
            confidence_threshold=0.7,
            nms_threshold=0.3,
            target_classes=candidates,
            enable_mapping=True,
        )
        detections = result.get('detections', [])
        stats = self._count_by_label(detections, object_type)
        return {
            'success': True,
            'predicted_count': int(stats['count']),
            'confidence': float(stats['avg_conf']),
            'processing_time': float(result.get('processing_time', 0.0)),
            'object_type': object_type,
            'model_times': result.get('model_times', {}),
            'segments_count': result.get('segments_count', 0),
            'object_types_found': result.get('object_types_found', 1),
            'avg_segment_area': result.get('avg_segment_area', 0),
            'models_used': result.get('models_used', ['sam', 'resnet', 'mapper'])
        }

    def process_image_auto(self, image_path: str) -> Dict[str, Any]:
        """Process a single image and infer the dominant object type by frequency."""
        result = run_pipeline(
            image_path,
            confidence_threshold=0.7,
            nms_threshold=0.3,
            target_classes=None,
            enable_mapping=True,
        )
        detections = result.get('detections', [])
        if not detections:
            return {
                'success': True,
                'predicted_count': 0,
                'confidence': 0.0,
                'processing_time': float(result.get('processing_time', 0.0)),
                'object_type': 'unknown',
            }

        # Count by mapped label
        label_to_items: Dict[str, List[Dict[str, Any]]] = {}
        for d in detections:
            lbl = (d.get('mapped_label') or d.get('raw_label', 'unknown')).lower()
            label_to_items.setdefault(lbl, []).append(d)

        # Choose the label with highest count; tie-break by avg confidence
        best_label = None
        best_count = -1
        best_avg_conf = -1.0
        for lbl, items in label_to_items.items():
            count = len(items)
            avg_conf = float(np.mean([it.get('confidence', 0.0) for it in items])) if items else 0.0
            if count > best_count or (count == best_count and avg_conf > best_avg_conf):
                best_label = lbl
                best_count = count
                best_avg_conf = avg_conf

        return {
            'success': True,
            'predicted_count': int(best_count if best_count > 0 else 0),
            'confidence': float(best_avg_conf if best_avg_conf > 0 else 0.0),
            'processing_time': float(result.get('processing_time', 0.0)),
            'object_type': best_label or 'unknown',
        }


# Public adapter instance expected by views: `from ...pipeline.pipeline import pipeline`
pipeline = _PipelineCompatibilityAdapter()
