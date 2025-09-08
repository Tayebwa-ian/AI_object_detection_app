"""
Label mapping and synonym handling
Maps raw model outputs to meaningful, consistent labels
"""

from typing import List, Dict, Any, Optional
import re
from transformers import pipeline


# Predefined synonym mappings
SYNONYM_MAPPINGS = {
    # People and animals
    'person': ['human', 'people', 'man', 'woman', 'child', 'boy', 'girl', 'adult'],
    'dog': ['puppy', 'canine', 'hound', 'mutt'],
    'cat': ['kitten', 'feline', 'kitty'],
    'bird': ['eagle', 'pigeon', 'sparrow', 'crow', 'seagull', 'hawk'],
    
    # Vehicles
    'car': ['automobile', 'sedan', 'hatchback', 'coupe', 'vehicle'],
    'truck': ['lorry', 'pickup', 'semi', 'trailer'],
    'bus': ['coach', 'transit'],
    'motorcycle': ['bike', 'motorbike', 'scooter'],
    'bicycle': ['bike', 'cycle'],
    
    # Objects
    'bottle': ['container', 'flask', 'jar'],
    'cup': ['mug', 'glass', 'tumbler'],
    'chair': ['seat', 'stool'],
    'table': ['desk', 'surface'],
    
    # Nature
    'tree': ['oak', 'pine', 'maple', 'palm', 'bush', 'shrub'],
    'flower': ['rose', 'tulip', 'daisy', 'bloom'],
    
    # Buildings and structures
    'building': ['house', 'structure', 'edifice', 'construction'],
    'road': ['street', 'path', 'highway', 'avenue'],
    'bridge': ['overpass', 'crossing'],
    
    # Technology
    'computer': ['laptop', 'pc', 'desktop'],
    'phone': ['smartphone', 'mobile', 'cellphone', 'iphone'],
    'tv': ['television', 'monitor', 'screen', 'display'],
}


# Reverse mapping for quick lookup
def build_reverse_mapping(synonym_dict: Dict[str, List[str]]) -> Dict[str, str]:
    """Build reverse mapping from synonym to canonical label"""
    reverse = {}
    for canonical, synonyms in synonym_dict.items():
        # Add canonical form
        reverse[canonical.lower()] = canonical
        
        # Add all synonyms
        for synonym in synonyms:
            reverse[synonym.lower()] = canonical
    
    return reverse


REVERSE_MAPPING = build_reverse_mapping(SYNONYM_MAPPINGS)


class LabelMapper:
    """Handles label mapping and synonym resolution"""
    
    def __init__(self, use_zero_shot: bool = True):
        """
        Initialize label mapper
        
        Args:
            use_zero_shot: Whether to use zero-shot classification for mapping
        """
        self.use_zero_shot = use_zero_shot
        self.zero_shot_classifier = None
        self._cache = {}  # Cache for zero-shot results
        
        if use_zero_shot:
            self._load_zero_shot_classifier()
    
    def _load_zero_shot_classifier(self):
        """Load zero-shot classification model"""
        try:
            self.zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="typeform/distilbert-base-uncased-mnli",
                device=-1  # CPU
            )
            print("Zero-shot classifier loaded")
        except Exception as e:
            print(f"Zero-shot classifier failed to load: {e}")
            self.use_zero_shot = False
    
    def map_synonym(self, raw_label: str) -> str:
        """
        Map raw label to canonical form using synonym dictionary
        
        Args:
            raw_label: Raw label from model
        
        Returns:
            Canonical label or original if no mapping found
        """
        # Clean the label
        cleaned = re.sub(r'[^a-zA-Z\s]', '', raw_label.lower().strip())
        
        # Direct lookup
        if cleaned in REVERSE_MAPPING:
            return REVERSE_MAPPING[cleaned]
        
        # Partial matching
        for synonym, canonical in REVERSE_MAPPING.items():
            if synonym in cleaned or cleaned in synonym:
                return canonical
        
        # Return original if no mapping found
        return raw_label
    
    def zero_shot_map(self, raw_label: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """
        Use zero-shot classification to map label to candidates
        
        Args:
            raw_label: Raw label from model
            candidate_labels: List of target labels
        
        Returns:
            Dictionary with mapped label and confidence
        """
        if not self.use_zero_shot or not self.zero_shot_classifier:
            return {'mapped_label': raw_label, 'mapping_confidence': 1.0}
        
        # Check cache
        cache_key = f"{raw_label}_{hash(tuple(candidate_labels))}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Run zero-shot classification
            result = self.zero_shot_classifier(raw_label, candidate_labels)
            
            mapped_label = result['labels'][0]
            mapping_confidence = result['scores'][0]
            
            # Cache result
            mapping_result = {
                'mapped_label': mapped_label,
                'mapping_confidence': mapping_confidence,
                'all_scores': dict(zip(result['labels'], result['scores']))
            }
            
            self._cache[cache_key] = mapping_result
            return mapping_result
            
        except Exception as e:
            print(f"Zero-shot mapping failed for '{raw_label}': {e}")
            return {'mapped_label': raw_label, 'mapping_confidence': 1.0}
    
    def map_label(self, raw_label: str, 
                  candidate_labels: Optional[List[str]] = None,
                  mapping_threshold: float = 0.5,
                  image_segment=None,
                  raw_confidence: float = 1.0) -> Dict[str, Any]:
        """
        FIXED: Complete label mapping pipeline with confidence-aware processing
        
        Args:
            raw_label: Raw label from model
            candidate_labels: Optional target labels for zero-shot
            mapping_threshold: Minimum confidence for zero-shot mapping
            image_segment: Optional image segment for visual validation
            raw_confidence: Confidence of the raw prediction
        
        Returns:
            Complete mapping result
        """
        result = {
            'raw_label': raw_label,
            'synonym_mapped': None,
            'zero_shot_mapped': None,
            'final_label': raw_label,
            'mapping_method': 'none',
            'mapping_confidence': raw_confidence,
            'mapping_applied': False
        }
        
        # FIXED: Step 1 - Check if raw prediction needs mapping
        if candidate_labels:
            direct_matches = [label for label in candidate_labels 
                             if label.lower() in raw_label.lower()]
            
            if direct_matches and raw_confidence > 0.8:
                # High confidence direct match - don't map
                result['final_label'] = direct_matches[0]
                result['mapping_method'] = 'direct_match'
                result['mapping_confidence'] = raw_confidence
                result['reason'] = 'High confidence direct match'
                return result
        
        # Step 2: Synonym mapping
        synonym_mapped = self.map_synonym(raw_label)
        result['synonym_mapped'] = synonym_mapped
        
        if synonym_mapped != raw_label:
            result['final_label'] = synonym_mapped
            result['mapping_method'] = 'synonym'
            result['mapping_confidence'] = 1.0
            result['mapping_applied'] = True
            return result
        
        # FIXED: Step 3 - Confidence-aware zero-shot mapping
        if candidate_labels and self.use_zero_shot:
            # Only apply zero-shot if original confidence isn't too high
            if raw_confidence < 0.9:  # Allow mapping for uncertain predictions
                zero_shot_result = self.zero_shot_map(raw_label, candidate_labels)
                result['zero_shot_mapped'] = zero_shot_result['mapped_label']
                
                # FIXED: Compare original vs mapped confidence
                mapping_confidence = zero_shot_result['mapping_confidence']
                
                if mapping_confidence >= mapping_threshold:
                    # Check if mapping improves confidence significantly
                    if mapping_confidence > raw_confidence + 0.1 or raw_confidence < 0.6:
                        result['final_label'] = zero_shot_result['mapped_label']
                        result['mapping_method'] = 'zero_shot'
                        result['mapping_confidence'] = mapping_confidence
                        result['mapping_applied'] = True
                        result['all_scores'] = zero_shot_result.get('all_scores', {})
                        result['reason'] = 'Beneficial zero-shot mapping'
                    else:
                        result['reason'] = 'Original prediction more confident'
                else:
                    result['reason'] = f'Mapping confidence too low ({mapping_confidence:.2f})'
            else:
                result['reason'] = 'Original prediction too confident for mapping'
        
        return result


# Global mapper instance
_global_mapper = None


def get_mapper(use_zero_shot: bool = True) -> LabelMapper:
    """Get global mapper instance (singleton pattern)"""
    global _global_mapper
    if _global_mapper is None:
        _global_mapper = LabelMapper(use_zero_shot=use_zero_shot)
    return _global_mapper


def map_labels(detections: List[Dict], 
               candidate_labels: Optional[List[str]] = None,
               mapping_threshold: float = 0.5) -> List[Dict]:
    """
    FIXED: Map labels for a list of detections with enhanced confidence handling
    
    Args:
        detections: List of detection results
        candidate_labels: Optional target labels
        mapping_threshold: Minimum confidence for zero-shot mapping
    
    Returns:
        Detections with mapped labels
    """
    mapper = get_mapper()
    mapped_detections = []
    
    for detection in detections:
        raw_label = detection.get('raw_label', 'unknown')
        raw_confidence = detection.get('calibrated_confidence', detection.get('confidence', 1.0))
        
        # FIXED: Map the label with confidence information
        mapping_result = mapper.map_label(
            raw_label, 
            candidate_labels, 
            mapping_threshold,
            image_segment=None,  # Could be enhanced with actual segment
            raw_confidence=raw_confidence
        )
        
        # Add mapping info to detection
        mapped_detection = detection.copy()
        mapped_detection.update({
            'mapped_label': mapping_result['final_label'],
            'mapping_method': mapping_result['mapping_method'],
            'mapping_confidence': mapping_result['mapping_confidence'],
            'mapping_applied': mapping_result.get('mapping_applied', False),
            'mapping_reason': mapping_result.get('reason', 'No mapping needed'),
            'mapping_details': mapping_result
        })
        
        mapped_detections.append(mapped_detection)
    
    return mapped_detections


def get_synonyms(label: str) -> List[str]:
    """Get all synonyms for a given label"""
    canonical = REVERSE_MAPPING.get(label.lower(), label)
    return SYNONYM_MAPPINGS.get(canonical, [])


def add_custom_synonyms(canonical_label: str, synonyms: List[str]):
    """Add custom synonym mapping"""
    global REVERSE_MAPPING
    
    if canonical_label not in SYNONYM_MAPPINGS:
        SYNONYM_MAPPINGS[canonical_label] = []
    
    # Add new synonyms
    for synonym in synonyms:
        if synonym not in SYNONYM_MAPPINGS[canonical_label]:
            SYNONYM_MAPPINGS[canonical_label].append(synonym)
            REVERSE_MAPPING[synonym.lower()] = canonical_label
    
    print(f"Added synonyms for '{canonical_label}': {synonyms}")


def get_all_canonical_labels() -> List[str]:
    """Get all canonical labels"""
    return list(SYNONYM_MAPPINGS.keys())


def create_label_hierarchy() -> Dict[str, List[str]]:
    """Create hierarchical label grouping"""
    hierarchy = {
        'living': ['person', 'dog', 'cat', 'bird'],
        'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
        'objects': ['bottle', 'cup', 'chair', 'table'],
        'nature': ['tree', 'flower'],
        'structures': ['building', 'road', 'bridge'],
        'technology': ['computer', 'phone', 'tv']
    }
    return hierarchy


def get_label_category(label: str) -> Optional[str]:
    """Get category for a given label"""
    hierarchy = create_label_hierarchy()
    
    for category, labels in hierarchy.items():
        if label.lower() in [l.lower() for l in labels]:
            return category
    
    return None


# Predefined candidate label sets for common use cases
CANDIDATE_SETS = {
    'general': ['person', 'car', 'dog', 'cat', 'bird', 'bicycle', 'motorcycle', 'bus', 'truck', 'tree', 'building'],
    'traffic': ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic_light', 'stop_sign', 'road'],
    'indoor': ['person', 'chair', 'table', 'cup', 'bottle', 'computer', 'phone', 'tv', 'book'],
    'outdoor': ['person', 'car', 'tree', 'building', 'dog', 'cat', 'bird', 'bicycle', 'road', 'sky'],
    'animals': ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe']
}


def get_candidate_set(set_name: str) -> List[str]:
    """Get predefined candidate label set"""
    return CANDIDATE_SETS.get(set_name, CANDIDATE_SETS['general'])


# Testing function
def test_mapping():
    """Test label mapping functionality"""
    test_labels = [
        'sports car', 'golden retriever', 'smartphone', 
        'oak tree', 'coffee mug', 'mountain bike'
    ]
    
    mapper = get_mapper()
    candidate_labels = get_candidate_set('general')
    
    print("Testing label mapping...")
    for label in test_labels:
        result = mapper.map_label(label, candidate_labels)
        print(f"'{label}' → '{result['final_label']}' (method: {result['mapping_method']})")


if __name__ == "__main__":
    test_mapping()
