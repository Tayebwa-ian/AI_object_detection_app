"""
Post-processing utilities
Filtering, thresholds, NMS, and result aggregation
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import cv2
import matplotlib.pyplot as plt
from collections import Counter

# Optional seaborn import for better styling
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    sns = None


def filter_segments(classifications: List[Dict], 
                   bboxes: List[List], 
                   confidence_threshold: float = 0.7,
                   min_area: int = 500,
                   max_area: int = None) -> List[Dict]:
    """
    Filter segments based on confidence and area
    
    Args:
        classifications: List of classification results
        bboxes: List of bounding boxes [x, y, w, h]
        confidence_threshold: Minimum confidence score
        min_area: Minimum segment area in pixels
        max_area: Maximum segment area in pixels (optional)
    
    Returns:
        Filtered list of results
    """
    filtered = []
    
    for i, (cls_result, bbox) in enumerate(zip(classifications, bboxes)):
        confidence = cls_result.get('confidence', 0.0)
        
        # Confidence filter
        if confidence < confidence_threshold:
            continue
        
        # Area filter
        x, y, w, h = bbox
        area = w * h
        
        if area < min_area:
            continue
        
        if max_area and area > max_area:
            continue
        
        # Add bbox info to result
        result = cls_result.copy()
        result['bbox'] = bbox
        result['area'] = area
        
        filtered.append(result)
    
    return filtered


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        box1, box2: [x, y, w, h] format
    
    Returns:
        IoU score between 0 and 1
    """
    # Convert to [x1, y1, x2, y2] format
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def apply_nms(detections: List[Dict], 
              threshold: float = 0.3,
              score_key: str = 'confidence') -> List[Dict]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections
    
    Args:
        detections: List of detection results with bbox and confidence
        threshold: IoU threshold for suppression
        score_key: Key for confidence score
    
    Returns:
        Filtered detections after NMS
    """
    if not detections:
        return []
    
    # Sort by confidence (descending)
    sorted_detections = sorted(detections, 
                             key=lambda x: x.get(score_key, 0), 
                             reverse=True)
    
    keep = []
    used = set()
    
    for i, detection in enumerate(sorted_detections):
        if i in used:
            continue
        
        keep.append(detection)
        current_bbox = detection['bbox']
        
        # Compare with remaining detections
        for j, other_detection in enumerate(sorted_detections[i+1:], i+1):
            if j in used:
                continue
            
            other_bbox = other_detection['bbox']
            iou = calculate_iou(current_bbox, other_bbox)
            
            # Suppress if IoU is above threshold
            if iou > threshold:
                used.add(j)
    
    return keep


def group_by_class(detections: List[Dict]) -> Dict[str, List[Dict]]:
    """Group detections by class label"""
    groups = {}
    
    for detection in detections:
        label = detection.get('mapped_label', detection.get('raw_label', 'unknown'))
        
        if label not in groups:
            groups[label] = []
        
        groups[label].append(detection)
    
    return groups


def calculate_class_statistics(grouped_detections: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Calculate statistics for each class"""
    stats = {}
    
    for class_name, detections in grouped_detections.items():
        confidences = [d.get('confidence', 0) for d in detections]
        areas = [d.get('area', 0) for d in detections]
        
        stats[class_name] = {
            'count': len(detections),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'total_area': sum(areas),
            'avg_area': np.mean(areas) if areas else 0
        }
    
    return stats


def aggregate_results(detections: List[Dict]) -> List[Dict]:
    """
    Aggregate and enrich detection results
    
    Args:
        detections: List of detection results
    
    Returns:
        Enriched results with additional metadata
    """
    if not detections:
        return []
    
    # Group by class
    grouped = group_by_class(detections)
    
    # Calculate statistics
    class_stats = calculate_class_statistics(grouped)
    
    # Enrich each detection
    enriched_detections = []
    
    for detection in detections:
        enriched = detection.copy()
        
        # Add class statistics
        label = detection.get('mapped_label', detection.get('raw_label', 'unknown'))
        if label in class_stats:
            enriched['class_stats'] = class_stats[label]
        
        # Add relative confidence (compared to class average)
        class_avg_conf = class_stats.get(label, {}).get('avg_confidence', 0)
        detection_conf = detection.get('confidence', 0)
        enriched['relative_confidence'] = detection_conf / class_avg_conf if class_avg_conf > 0 else 1.0
        
        # Convert bbox to different formats for convenience
        if 'bbox' in enriched:
            x, y, w, h = enriched['bbox']
            enriched['bbox_formats'] = {
                'xywh': [x, y, w, h],
                'xyxy': [x, y, x + w, y + h],
                'center': [x + w/2, y + h/2, w, h]
            }
        
        enriched_detections.append(enriched)
    
    return enriched_detections


def apply_confidence_boost(detections: List[Dict], 
                          boost_classes: List[str] = None,
                          boost_factor: float = 1.2) -> List[Dict]:
    """
    Boost confidence for specific classes (useful for important objects)
    
    Args:
        detections: List of detection results
        boost_classes: Classes to boost (e.g., ['person', 'car'])
        boost_factor: Multiplication factor for confidence
    
    Returns:
        Detections with boosted confidence scores
    """
    if not boost_classes:
        return detections
    
    boosted = []
    
    for detection in detections:
        result = detection.copy()
        label = detection.get('mapped_label', detection.get('raw_label', 'unknown'))
        
        if label.lower() in [cls.lower() for cls in boost_classes]:
            original_conf = result.get('confidence', 0)
            boosted_conf = min(1.0, original_conf * boost_factor)  # Cap at 1.0
            result['confidence'] = boosted_conf
            result['confidence_boosted'] = True
        
        boosted.append(result)
    
    return boosted


def create_summary_report(detections: List[Dict], 
                         original_segments_count: int = 0) -> Dict[str, Any]:
    """
    Create a comprehensive summary report
    
    Args:
        detections: Final detection results
        original_segments_count: Number of segments before filtering
    
    Returns:
        Summary statistics and insights
    """
    if not detections:
        return {
            'total_objects': 0,
            'classes_detected': [],
            'processing_efficiency': 0,
            'quality_score': 0
        }
    
    # Group by class
    grouped = group_by_class(detections)
    class_stats = calculate_class_statistics(grouped)
    
    # Calculate metrics
    total_objects = len(detections)
    unique_classes = len(grouped)
    avg_confidence = np.mean([d.get('confidence', 0) for d in detections])
    
    # Processing efficiency
    efficiency = (total_objects / original_segments_count * 100) if original_segments_count > 0 else 0
    
    # Quality score (based on confidence distribution)
    confidences = [d.get('confidence', 0) for d in detections]
    high_conf_count = sum(1 for c in confidences if c > 0.8)
    quality_score = (high_conf_count / total_objects * 100) if total_objects > 0 else 0
    
    return {
        'total_objects': total_objects,
        'unique_classes': unique_classes,
        'classes_detected': list(grouped.keys()),
        'class_distribution': {k: len(v) for k, v in grouped.items()},
        'avg_confidence': round(avg_confidence, 3),
        'confidence_distribution': {
            'high (>0.8)': sum(1 for c in confidences if c > 0.8),
            'medium (0.5-0.8)': sum(1 for c in confidences if 0.5 <= c <= 0.8),
            'low (<0.5)': sum(1 for c in confidences if c < 0.5)
        },
        'processing_efficiency': round(efficiency, 1),
        'quality_score': round(quality_score, 1),
        'class_statistics': class_stats
    }


def apply_confidence_thresholds(detections: List[Dict], 
                               thresholds: Dict[str, float] = None,
                               default_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Apply different confidence thresholds and return filtered results with counts
    
    Args:
        detections: List of detection results
        thresholds: Dict of class-specific thresholds {'person': 0.8, 'car': 0.6}
        default_threshold: Default threshold for classes not specified
    
    Returns:
        Dict with filtered results and threshold analysis
    """
    if not detections:
        return {
            'filtered_detections': [],
            'threshold_analysis': {},
            'total_passed': 0,
            'total_filtered': 0
        }
    
    if thresholds is None:
        thresholds = {}
    
    passed_detections = []
    filtered_detections = []
    threshold_analysis = {}
    
    for detection in detections:
        label = detection.get('mapped_label', detection.get('raw_label', 'unknown'))
        confidence = detection.get('confidence', 0.0)
        
        # Get threshold for this class
        threshold = thresholds.get(label, default_threshold)
        
        # Track threshold analysis
        if label not in threshold_analysis:
            threshold_analysis[label] = {
                'threshold_used': threshold,
                'total_detections': 0,
                'passed': 0,
                'filtered': 0,
                'avg_confidence': 0,
                'confidences': []
            }
        
        threshold_analysis[label]['total_detections'] += 1
        threshold_analysis[label]['confidences'].append(confidence)
        
        if confidence >= threshold:
            passed_detections.append(detection)
            threshold_analysis[label]['passed'] += 1
        else:
            filtered_detections.append(detection)
            threshold_analysis[label]['filtered'] += 1
    
    # Calculate averages
    for label, analysis in threshold_analysis.items():
        if analysis['confidences']:
            analysis['avg_confidence'] = np.mean(analysis['confidences'])
    
    return {
        'filtered_detections': passed_detections,
        'failed_detections': filtered_detections,
        'threshold_analysis': threshold_analysis,
        'total_passed': len(passed_detections),
        'total_filtered': len(filtered_detections)
    }


def count_objects_by_class(detections: List[Dict]) -> Dict[str, Dict[str, Any]]:
    """
    Count objects by class with detailed statistics
    
    Args:
        detections: List of detection results
    
    Returns:
        Comprehensive object counts and statistics
    """
    if not detections:
        return {}
    
    # Group by class
    grouped = group_by_class(detections)
    counts = {}
    
    for class_name, class_detections in grouped.items():
        confidences = [d.get('confidence', 0) for d in class_detections]
        areas = [d.get('area', 0) for d in class_detections]
        
        counts[class_name] = {
            'count': len(class_detections),
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences),
                'median': np.median(confidences)
            },
            'area_stats': {
                'mean': np.mean(areas) if areas else 0,
                'std': np.std(areas) if areas else 0,
                'min': np.min(areas) if areas else 0,
                'max': np.max(areas) if areas else 0,
                'total': np.sum(areas) if areas else 0
            },
            'quality_flags': {
                'high_confidence': sum(1 for c in confidences if c > 0.8),
                'medium_confidence': sum(1 for c in confidences if 0.5 <= c <= 0.8),
                'low_confidence': sum(1 for c in confidences if c < 0.5),
                'reliable': sum(1 for c in confidences if c > 0.7) / len(confidences) > 0.5
            }
        }
    
    return counts


def aggregate_counts_across_images(image_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Aggregate object counts across multiple images
    
    Args:
        image_results: Dict mapping image names to their detection results
    
    Returns:
        Aggregated statistics across all images
    """
    if not image_results:
        return {}
    
    all_detections = []
    per_image_stats = {}
    
    # Collect all detections and per-image stats
    for image_name, detections in image_results.items():
        all_detections.extend(detections)
        per_image_stats[image_name] = count_objects_by_class(detections)
    
    # Overall statistics
    overall_counts = count_objects_by_class(all_detections)
    
    # Cross-image analysis
    all_classes = set()
    for stats in per_image_stats.values():
        all_classes.update(stats.keys())
    
    cross_image_analysis = {}
    for class_name in all_classes:
        class_counts = []
        class_confidences = []
        
        for image_name, stats in per_image_stats.items():
            if class_name in stats:
                class_counts.append(stats[class_name]['count'])
                class_confidences.extend([
                    stats[class_name]['confidence_stats']['mean']
                ])
        
        cross_image_analysis[class_name] = {
            'appears_in_images': len(class_counts),
            'total_images': len(per_image_stats),
            'frequency': len(class_counts) / len(per_image_stats),
            'count_variance': np.var(class_counts) if len(class_counts) > 1 else 0,
            'avg_count_per_image': np.mean(class_counts),
            'consistency_score': 1 - (np.std(class_counts) / np.mean(class_counts)) if np.mean(class_counts) > 0 else 0
        }
    
    return {
        'overall_counts': overall_counts,
        'per_image_stats': per_image_stats,
        'cross_image_analysis': cross_image_analysis,
        'summary': {
            'total_images': len(per_image_stats),
            'total_objects': len(all_detections),
            'unique_classes': len(all_classes),
            'avg_objects_per_image': len(all_detections) / len(per_image_stats) if per_image_stats else 0
        }
    }


def generate_histograms(detections: List[Dict], 
                       save_path: str = None,
                       show_plots: bool = True) -> Dict[str, Any]:
    """
    Generate comprehensive histograms for object detection results
    
    Args:
        detections: List of detection results
        save_path: Path to save histogram plots (optional)
        show_plots: Whether to display plots
    
    Returns:
        Dict containing histogram data and plot information
    """
    if not detections:
        return {'error': 'No detections to plot'}
    
    # Extract data
    classes = [d.get('mapped_label', d.get('raw_label', 'unknown')) for d in detections]
    confidences = [d.get('confidence', 0) for d in detections]
    areas = [d.get('area', 0) for d in detections if d.get('area', 0) > 0]
    
    # Class distribution histogram
    class_counts = Counter(classes)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Object Detection Analysis Histograms', fontsize=16, fontweight='bold')
    
    # 1. Class Distribution
    ax1 = axes[0, 0]
    classes_list = list(class_counts.keys())
    counts_list = list(class_counts.values())
    
    bars = ax1.bar(classes_list, counts_list, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_title('Object Count by Class', fontweight='bold')
    ax1.set_xlabel('Object Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts_list):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Confidence Distribution
    ax2 = axes[0, 1]
    ax2.hist(confidences, bins=20, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    ax2.set_title('Confidence Score Distribution', fontweight='bold')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Area Distribution (if available)
    ax3 = axes[1, 0]
    if areas:
        ax3.hist(areas, bins=20, color='orange', edgecolor='darkorange', alpha=0.7)
        ax3.axvline(np.mean(areas), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(areas):.0f}px²')
        ax3.set_title('Object Area Distribution', fontweight='bold')
        ax3.set_xlabel('Area (pixels²)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No area data available', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Object Area Distribution', fontweight='bold')
    
    # 4. Confidence by Class (Box Plot)
    ax4 = axes[1, 1]
    
    # Prepare data for box plot
    class_confidence_data = []
    class_labels = []
    
    for class_name in class_counts.keys():
        class_confidences = [d.get('confidence', 0) for d in detections 
                           if d.get('mapped_label', d.get('raw_label', 'unknown')) == class_name]
        if class_confidences:
            class_confidence_data.append(class_confidences)
            class_labels.append(class_name)
    
    if class_confidence_data:
        bp = ax4.boxplot(class_confidence_data, labels=class_labels, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_title('Confidence Distribution by Class', fontweight='bold')
        ax4.set_xlabel('Object Class')
        ax4.set_ylabel('Confidence Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No confidence data available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Confidence Distribution by Class', fontweight='bold')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Histograms saved to: {save_path}")
    
    # Show if requested
    if show_plots:
        plt.show()
    
    # Return histogram data
    histogram_data = {
        'class_distribution': dict(class_counts),
        'confidence_stats': {
            'mean': np.mean(confidences),
            'std': np.std(confidences),
            'min': np.min(confidences),
            'max': np.max(confidences),
            'median': np.median(confidences),
            'histogram_bins': np.histogram(confidences, bins=20)
        },
        'area_stats': {
            'mean': np.mean(areas) if areas else 0,
            'std': np.std(areas) if areas else 0,
            'min': np.min(areas) if areas else 0,
            'max': np.max(areas) if areas else 0,
            'histogram_bins': np.histogram(areas, bins=20) if areas else None
        },
        'total_objects': len(detections),
        'unique_classes': len(class_counts)
    }
    
    return histogram_data


def generate_quality_flags(detections: List[Dict], 
                          confidence_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Generate quality flags and assessment for detection results
    
    Args:
        detections: List of detection results
        confidence_thresholds: Class-specific confidence thresholds
    
    Returns:
        Quality assessment with flags and recommendations
    """
    if not detections:
        return {'overall_quality': 'NO_DETECTIONS', 'flags': [], 'recommendations': []}
    
    if confidence_thresholds is None:
        confidence_thresholds = {}
    
    flags = []
    recommendations = []
    quality_metrics = {}
    
    # Extract data
    confidences = [d.get('confidence', 0) for d in detections]
    classes = [d.get('mapped_label', d.get('raw_label', 'unknown')) for d in detections]
    areas = [d.get('area', 0) for d in detections if d.get('area', 0) > 0]
    
    # 1. Confidence Quality Assessment
    high_conf_count = sum(1 for c in confidences if c > 0.8)
    medium_conf_count = sum(1 for c in confidences if 0.5 <= c <= 0.8)
    low_conf_count = sum(1 for c in confidences if c < 0.5)
    
    confidence_quality = high_conf_count / len(confidences)
    quality_metrics['confidence_quality'] = confidence_quality
    
    if confidence_quality > 0.7:
        flags.append('HIGH_CONFIDENCE_DETECTIONS')
    elif confidence_quality < 0.3:
        flags.append('LOW_CONFIDENCE_DETECTIONS')
        recommendations.append('Consider lowering confidence thresholds or improving model')
    
    # 2. Class Distribution Assessment
    class_counts = Counter(classes)
    max_class_count = max(class_counts.values())
    min_class_count = min(class_counts.values())
    
    if max_class_count / min_class_count > 5:
        flags.append('IMBALANCED_CLASS_DISTRIBUTION')
        recommendations.append('Class distribution is highly imbalanced')
    
    # 3. Detection Density Assessment
    total_detections = len(detections)
    if total_detections > 50:
        flags.append('HIGH_DETECTION_DENSITY')
        recommendations.append('Consider applying stricter filtering or NMS')
    elif total_detections < 5:
        flags.append('LOW_DETECTION_DENSITY')
        recommendations.append('Consider lowering confidence thresholds')
    
    # 4. Area Distribution Assessment
    if areas:
        area_std = np.std(areas)
        area_mean = np.mean(areas)
        if area_std / area_mean > 2:  # High coefficient of variation
            flags.append('HIGH_SIZE_VARIATION')
        
        # Check for very small or very large objects
        very_small = sum(1 for a in areas if a < 500)
        very_large = sum(1 for a in areas if a > 50000)
        
        if very_small / len(areas) > 0.3:
            flags.append('MANY_SMALL_OBJECTS')
            recommendations.append('Consider increasing minimum area threshold')
        
        if very_large / len(areas) > 0.2:
            flags.append('MANY_LARGE_OBJECTS')
            recommendations.append('Large objects detected - verify they are not background')
    
    # 5. Overall Quality Score
    quality_score = (
        confidence_quality * 0.4 +  # 40% weight on confidence
        min(1.0, total_detections / 20) * 0.3 +  # 30% weight on detection count (capped)
        (1 - min(1.0, len(class_counts) / 10)) * 0.3  # 30% weight on class diversity
    )
    
    quality_metrics['overall_score'] = quality_score
    
    # Determine overall quality level
    if quality_score > 0.8:
        overall_quality = 'EXCELLENT'
    elif quality_score > 0.6:
        overall_quality = 'GOOD'
    elif quality_score > 0.4:
        overall_quality = 'FAIR'
    else:
        overall_quality = 'POOR'
        recommendations.append('Consider reviewing detection pipeline parameters')
    
    return {
        'overall_quality': overall_quality,
        'quality_score': round(quality_score, 3),
        'quality_metrics': quality_metrics,
        'flags': flags,
        'recommendations': recommendations,
        'statistics': {
            'total_detections': total_detections,
            'confidence_distribution': {
                'high': high_conf_count,
                'medium': medium_conf_count,
                'low': low_conf_count
            },
            'class_distribution': dict(class_counts),
            'area_stats': {
                'mean': np.mean(areas) if areas else 0,
                'std': np.std(areas) if areas else 0,
                'count': len(areas)
            }
        }
    }
