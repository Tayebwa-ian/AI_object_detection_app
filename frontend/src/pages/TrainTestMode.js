// src/pages/TrainTestMode.js

import React, { useState } from "react";
import { Box, Typography, Card, CardContent, TextField, MenuItem, Grid, Button } from "@mui/material";
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
import { dummyTrainingMetrics } from "../data/dummy";

// stage models (can be replaced by getModels if you wire it)
const stageModels = {
  segmentation: ["SAM", "Deeplabv3"],
  feature_extraction: ["ResNet50", "EfficientNet"],
  classification: ["Logistic Regression", "Linear Probe"],
};

const TrainTestMode = () => {
  const [selectedModels, setSelectedModels] = useState({
    segmentation: "SAM",
    feature_extraction: "ResNet50",
    classification: "Logistic Regression",
  });

  const [trainingSamples, setTrainingSamples] = useState(100);
  const [testingSamples, setTestingSamples] = useState(20);

  const handleModelChange = (stage, value) => {
    setSelectedModels((prev) => ({ ...prev, [stage]: value }));
  };

  // Use the unified shape from dummyTrainingMetrics
  const { overall, latency, confusion_matrix } = dummyTrainingMetrics;

  const overallData = [
    { metric: "Accuracy", value: overall.accuracy },
    { metric: "Precision", value: overall.precision },
    { metric: "Recall", value: overall.recall },
    { metric: "F1 Score", value: overall.f1_score },
  ];

  const latencyData = Object.entries(latency).map(([stage, value]) => ({ stage, value }));

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Train & Test Mode
      </Typography>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <TextField
            select
            fullWidth
            label="Segmentation Model"
            value={selectedModels.segmentation}
            onChange={(e) => handleModelChange("segmentation", e.target.value)}
          >
            {stageModels.segmentation.map((m) => <MenuItem key={m} value={m}>{m}</MenuItem>)}
          </TextField>
        </Grid>

        <Grid item xs={12} sm={4}>
          <TextField
            select
            fullWidth
            label="Feature Extraction Model"
            value={selectedModels.feature_extraction}
            onChange={(e) => handleModelChange("feature_extraction", e.target.value)}
          >
            {stageModels.feature_extraction.map((m) => <MenuItem key={m} value={m}>{m}</MenuItem>)}
          </TextField>
        </Grid>

        <Grid item xs={12} sm={4}>
          <TextField
            select
            fullWidth
            label="Classification Model"
            value={selectedModels.classification}
            onChange={(e) => handleModelChange("classification", e.target.value)}
          >
            {stageModels.classification.map((m) => <MenuItem key={m} value={m}>{m}</MenuItem>)}
          </TextField>
        </Grid>
      </Grid>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            type="number"
            label="Number of Training Samples"
            value={trainingSamples}
            onChange={(e) => setTrainingSamples(Number(e.target.value))}
          />
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            type="number"
            label="Number of Testing Samples"
            value={testingSamples}
            onChange={(e) => setTestingSamples(Number(e.target.value))}
          />
        </Grid>
      </Grid>

      <Button variant="contained" sx={{ mb: 3 }}>
        Simulate Training Session
      </Button>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6">Overall Metrics</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={overallData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(value) => (value * 100).toFixed(2) + "%"} />
              <Bar dataKey="value" fill="#1976d2" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6">Latency per Stage (ms)</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={latencyData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="stage" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#ff9800" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6">Confusion Matrix (Dummy)</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={confusion_matrix} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#4caf50" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default TrainTestMode;
