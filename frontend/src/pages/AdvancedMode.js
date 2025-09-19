// src/pages/AdvancedMode.js
import React, { useState, useEffect, useCallback } from "react";
import { Typography, Button, Box, TextField, MenuItem } from "@mui/material";
import FileUpload from "../components/FileUpload";
import PredictionCard from "../components/PredictionCard";
import Feedback from "../components/Feedback";
import { useLoading } from "../context/LoadingContext";
import { getLabels } from "../api/labels";
import { getPrediction, saveCorrectedCount } from "../api/outputs";

const AdvancedMode = () => {
  const { startLoading, stopLoading } = useLoading();
  const [files, setFiles] = useState([]);
  const [labels, setLabels] = useState([]);
  const [selectedLabelId, setSelectedLabelId] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [feedback, setFeedback] = useState({ open: false, message: "", severity: "success" });

  const fetchLabels = useCallback(async () => {
    try {
      startLoading();
      const data = await getLabels();
      setLabels(data);
    } catch {
      setFeedback({ open: true, message: "Failed to load labels", severity: "error" });
    } finally {
      stopLoading();
    }
  }, [startLoading, stopLoading]);

  useEffect(() => { fetchLabels(); }, [fetchLabels]);

  const handleSubmit = async () => {
    if (files.length === 0 || !selectedLabelId) { setFeedback({ open: true, message: "Please upload images and select a label", severity: "warning" }); return; }
    try {
      startLoading();
      const labelObj = labels.find((l) => l.id === selectedLabelId);
      const labelName = labelObj ? labelObj.name : selectedLabelId;
      const results = [];
      for (let i = 0; i < files.length; i += 1) {
        // eslint-disable-next-line no-await-in-loop
        const pred = await getPrediction({ input_file: files[i], label_id: labelName });
        results.push(pred);
      }
      setPredictions(results);
      setFeedback({ open: true, message: "Predictions ready", severity: "success" });
    } catch (err) {
      console.error(err);
      setFeedback({ open: true, message: "Error generating predictions", severity: "error" });
    } finally {
      stopLoading();
    }
  };

  const handleSaveCorrected = async (predictionId, label, correctedCount) => {
    try {
      startLoading();
      await saveCorrectedCount({ output_id: predictionId, label, corrected_count: correctedCount });
      setPredictions((prev) =>
        prev.map((p) => (p.id === predictionId ? { ...p, detected_labels: p.detected_labels.map((dl) => (dl.label === label ? { ...dl, corrected_count: correctedCount } : dl)) } : p))
      );
      setFeedback({ open: true, message: `Corrected count saved for ${label}`, severity: "success" });
    } catch (err) {
      console.error(err);
      setFeedback({ open: true, message: "Error saving corrected count", severity: "error" });
    } finally {
      stopLoading();
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>Advanced Mode</Typography>

      <FileUpload files={files} setFiles={setFiles} />

      <Box sx={{ mt: 2 }}>
        <TextField select label="Label of Interest" value={selectedLabelId} onChange={(e) => setSelectedLabelId(e.target.value)} fullWidth>
          <MenuItem value="">-- select label --</MenuItem>
          {labels.map((l) => (<MenuItem key={l.id} value={l.id}>{l.name}</MenuItem>))}
        </TextField>
      </Box>

      <Box sx={{ mt: 2 }}>
        <Button variant="contained" onClick={handleSubmit} disabled={!files.length || !selectedLabelId}>Count</Button>
      </Box>

      <Box sx={{ mt: 3 }}>
        {predictions.map((p) => (
          <PredictionCard key={p.id} prediction={p} onSaveCorrected={(predId, label, corrected) => handleSaveCorrected(predId, label, corrected)} />
        ))}
      </Box>

      <Feedback open={feedback.open} message={feedback.message} severity={feedback.severity} onClose={() => setFeedback({ ...feedback, open: false })} />
    </Box>
  );
};

export default AdvancedMode;
