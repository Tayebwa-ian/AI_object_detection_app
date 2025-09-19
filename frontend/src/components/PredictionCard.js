// src/components/PredictionCard.js

import React, { useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Button,
  Divider,
  Chip,
} from "@mui/material";

/**
 * prediction object shape (generateFullDummyPrediction):
 * {
 *   id,
 *   image,
 *   label_of_interest,
 *   candidate_labels,
 *   predicted_count,
 *   detected_labels: [{ label, count, avg_confidence, num_segments, segments: [{id,label,count,confidence,image}...] }],
 *   segments: [...],
 *   models_used: { segmentation, feature_extraction, classification },
 *   latencies: { segmentation, feature_extraction, classification },
 *   timestamp
 * }
 *
 * onSaveCorrected(predictionId, label, correctedCount) -> function passed from parent
 */

const PredictionCard = ({ prediction, onSaveCorrected }) => {
  const [correctedMap, setCorrectedMap] = useState({}); // { label: correctedCount }

  if (!prediction) return null;

  const {
    image,
    label_of_interest,
    predicted_count,
    detected_labels,
    models_used,
    latencies,
    timestamp,
  } = prediction;

  const handleChange = (label, value) => {
    setCorrectedMap((m) => ({ ...m, [label]: value }));
  };

  const handleSave = (label) => {
    const value = Number(correctedMap[label]);
    if (!Number.isFinite(value)) return;
    if (onSaveCorrected) onSaveCorrected(prediction.id, label, value);
  };

  return (
    <Card sx={{ mt: 3 }}>
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <img
              src={image}
              alt="prediction"
              style={{ width: 256, height: 256, objectFit: "cover", borderRadius: 6 }}
            />
            <Box sx={{ mt: 1 }}>
              <Typography variant="subtitle2">Timestamp</Typography>
              <Typography variant="body2">{new Date(timestamp).toLocaleString()}</Typography>
            </Box>
            <Box sx={{ mt: 1 }}>
              <Typography variant="subtitle2">Models used</Typography>
              <Typography variant="body2">Segmentation: {models_used.segmentation}</Typography>
              <Typography variant="body2">Feature: {models_used.feature_extraction}</Typography>
              <Typography variant="body2">Classification: {models_used.classification}</Typography>
            </Box>
            <Box sx={{ mt: 1 }}>
              <Typography variant="subtitle2">Latencies (ms)</Typography>
              <Typography variant="body2">Seg: {latencies.segmentation} ms</Typography>
              <Typography variant="body2">Feat: {latencies.feature_extraction} ms</Typography>
              <Typography variant="body2">Cls: {latencies.classification} ms</Typography>
            </Box>
          </Grid>

          <Grid item xs={12} md={8}>
            <Typography variant="h6">Label of interest: {label_of_interest}</Typography>
            <Typography variant="subtitle1">Predicted total count: {predicted_count}</Typography>

            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2">Detected labels (aggregated)</Typography>
              <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mt: 1 }}>
                {detected_labels.map((lbl) => (
                  <Chip
                    key={lbl.label}
                    label={`${lbl.label} — ${lbl.count} (${Math.round(lbl.avg_confidence * 100)}%)`}
                    color={lbl.label === label_of_interest ? "primary" : "default"}
                    variant={lbl.label === label_of_interest ? "filled" : "outlined"}
                    sx={{ mb: 1 }}
                  />
                ))}
              </Box>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Box>
              <Typography variant="subtitle2">Corrected count for label of interest</Typography>
              <Box sx={{ display: "flex", gap: 1, alignItems: "center", mt: 1 }}>
                <TextField
                  size="small"
                  type="number"
                  label={`Correct ${label_of_interest}`}
                  value={correctedMap[label_of_interest] || ""}
                  onChange={(e) => handleChange(label_of_interest, e.target.value)}
                />
                <Button variant="outlined" onClick={() => handleSave(label_of_interest)}>Save</Button>
              </Box>
            </Box>

            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle2">Segments ({prediction.segments.length})</Typography>

            <Grid container spacing={2} sx={{ mt: 1 }}>
              {prediction.segments.map((seg) => (
                <Grid item key={seg.id} xs={12} sm={6} md={4}>
                  <Card variant="outlined">
                    <Box sx={{ p: 1 }}>
                      <img src={seg.image} alt={seg.label} style={{ width: "100%", height: 140, objectFit: "cover", borderRadius: 4 }} />
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="body2"><strong>Seg ID:</strong> {seg.id}</Typography>
                        <Typography variant="body2"><strong>Label:</strong> {seg.label}</Typography>
                        <Typography variant="body2"><strong>Count:</strong> {seg.count}</Typography>
                        <Typography variant="body2"><strong>Conf:</strong> {(seg.confidence * 100).toFixed(1)}%</Typography>
                      </Box>
                    </Box>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default PredictionCard;
