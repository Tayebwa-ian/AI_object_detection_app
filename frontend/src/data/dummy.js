// src/dummy.js

// ----------------- Your original content (kept exactly) -----------------
// Dummy images (can use public/dummy_images folder)
export const dummyImages = [
  "/dummy_images/car.jpg",
  "/dummy_images/dog.jpeg",
];

export const dummyLabels = [
  { id: "1", name: "Car", description: "Vehicle with 4 wheels" },
  { id: "2", name: "Dog", description: "Domestic animal" },
  { id: "3", name: "Tree", description: "Green plant" },
];


export const dummyPrediction = (file, labelOfInterest, candidateLabels = []) => ({
  id: "p1",
  image: URL.createObjectURL(file),
  predicted_count: Math.floor(Math.random() * 10) + 1,
  corrected_count: Math.floor(Math.random() * 10) + 1,
  label: labelOfInterest,
  candidate_labels: candidateLabels,
  detected_objects: candidateLabels.map((lbl) => ({
    label: lbl,
    count: Math.floor(Math.random() * 5),
  })),
});

// Generate dummy predictions
export const generateDummyPrediction = (file) => {
  const imageUrl = file ? URL.createObjectURL(file) : dummyImages[Math.floor(Math.random() * dummyImages.length)];

  const detectedObjects = Array.from({ length: Math.floor(Math.random() * 3) + 1 }, (_, idx) => ({
    id: `obj-${idx}`,
    label: `Object-${idx + 1}`,
    count: Math.floor(Math.random() * 5) + 1,
    confidence: Math.round(Math.random() * 100) / 100, // 0–1
    image: imageUrl,
  }));

  const predictedCount = detectedObjects.reduce((sum, obj) => sum + obj.count, 0);
  const correctedCount = detectedObjects.reduce((sum, obj) => sum + obj.count, 0);

  return {
    id: file ? file.name : `dummy-${Date.now()}`,
    image: imageUrl,
    predicted_count: predictedCount,
    corrected_count: correctedCount,

    label: "LabelOfInterest",
    detected_objects: detectedObjects,
  };
};
// ---------------------------------------------------------------------------


// ----------------- Additions: dummyModels and unified dummyTrainingMetrics ---------------

// Dummy Models (so ModelsPage and dropdowns can use them)
export const dummyModels = [
  { id: "m1", name: "SAM", description: "Segmentation: SAM" },
  { id: "m2", name: "DeepLabv3", description: "Segmentation: DeepLabv3" },
  { id: "m3", name: "ResNet50", description: "Feature extractor: ResNet50" },
  { id: "m4", name: "EfficientNet", description: "Feature extractor: EfficientNet" },
  { id: "m5", name: "Logistic Regression", description: "Classifier: Logistic Regression" },
  { id: "m6", name: "Linear Probe", description: "Classifier: Linear Probe" },
];

// Unified dummy training metrics (structure expected by TrainTestMode)
export const dummyTrainingMetrics = {
  overall: {
    accuracy: 0.72,
    precision: 0.68,
    recall: 0.60,
    f1_score: 0.64,
  },
  latency: {
    segmentation: 380, // ms
    feature_extraction: 103,
    classification: 120,
  },
  confusion_matrix: [
    { name: "Cat", value: 50 },
    { name: "Car", value: 30 },
    { name: "Phone", value: 20 },
  ],
};

// generateFullDummyPrediction: detailed structure per image (segments, labels, latencies, etc.)
export const generateFullDummyPrediction = ({
  file = null,
  labelOfInterest = "LabelOfInterest",
  candidateLabels = [],
  modelsUsed = {
    segmentation: "SAM",
    feature_extraction: "ResNet50",
    classification: "Logistic Regression",
  },
} = {}) => {
  const imageUrl = file
    ? (file instanceof File ? URL.createObjectURL(file) : file.url || file)
    : dummyImages[Math.floor(Math.random() * dummyImages.length)];

  const availableLabels = [
    ...candidateLabels,
    ...dummyLabels.map((d) => d.name),
  ].filter(Boolean);

  const labelPool = Array.from(new Set([labelOfInterest, ...availableLabels])).slice(0, 4);

  const segments = [];
  const numSegments = Math.max(1, Math.floor(Math.random() * 6)); // 1..6 segments
  for (let i = 0; i < numSegments; i += 1) {
    const label = labelPool[Math.floor(Math.random() * labelPool.length)];
    const count = Math.floor(Math.random() * 3) + 1;
    const confidence = Math.round((Math.random() * 0.5 + 0.5) * 100) / 100; // 0.5 - 1.0
    segments.push({
      id: `${file ? (file.name || file) : "dummy"}-seg-${i}-${Date.now()}`,
      label,
      count,
      confidence,
      image: imageUrl,
    });
  }

  const labelMap = {};
  for (const seg of segments) {
    if (!labelMap[seg.label]) labelMap[seg.label] = { label: seg.label, count: 0, confidences: [], segments: [] };
    labelMap[seg.label].count += seg.count;
    labelMap[seg.label].confidences.push(seg.confidence);
    labelMap[seg.label].segments.push(seg);
  }

  const detected_labels = Object.values(labelMap).map((item) => ({
    label: item.label,
    count: item.count,
    avg_confidence: Math.round((item.confidences.reduce((a, b) => a + b, 0) / item.confidences.length) * 100) / 100,
    num_segments: item.segments.length,
    segments: item.segments,
  }));

  const predicted_count = detected_labels.reduce((s, l) => s + l.count, 0);
  const corrected_count = detected_labels.reduce((s, l) => s + l.count, 0);

  const latencies = {
    segmentation: Math.round(80 + Math.random() * 300),
    feature_extraction: Math.round(40 + Math.random() * 100),
    classification: Math.round(30 + Math.random() * 100),
  };

  return {
    id: file ? (file.name || `${Date.now()}`) : `dummy-${Date.now()}`,
    image: imageUrl,
    label_of_interest: labelOfInterest,
    candidate_labels: candidateLabels,
    predicted_count,
    corrected_count,
    detected_labels,
    segments,
    models_used: modelsUsed,
    latencies,
    timestamp: new Date().toISOString(),
  };
}

export const generateFullDummyPredictionsForFiles = ({ files = [], labelOfInterest = "LabelOfInterest", candidateLabels = [] } = {}) =>
  files.map((f) => generateFullDummyPrediction({ file: f, labelOfInterest, candidateLabels }));
