// src/pages/NormalMode.js
import React, { useState, useEffect, useCallback } from "react";
import { Typography, Button, TextField, Box, Chip, MenuItem } from "@mui/material";
import FileUpload from "../components/FileUpload";
import Feedback from "../components/Feedback";
import PredictionCard from "../components/PredictionCard";
import { useLoading } from "../context/LoadingContext";
import { getLabels } from "../api/labels";
import { getPrediction, saveCorrectedCount } from "../api/outputs";

const NormalMode = () => {
  const { startLoading, stopLoading } = useLoading();
  const [files, setFiles] = useState([]);
  const [labels, setLabels] = useState([]);
  const [selectedLabelId, setSelectedLabelId] = useState("");
  const [candidateInput, setCandidateInput] = useState("");
  const [candidateLabels, setCandidateLabels] = useState([]);
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

  const handleAddCandidate = () => {
    if (candidateLabels.length >= 5) { setFeedback({ open: true, message: "Max 5 candidate labels", severity: "warning" }); return; }
    if (candidateInput.trim()) { setCandidateLabels((s) => [...s, candidateInput.trim()]); setCandidateInput(""); }
  };
  const handleRemoveCandidate = (label) => setCandidateLabels((s) => s.filter((l) => l !== label));

  const handleSubmit = async () => {
    if (files.length === 0 || !selectedLabelId) { setFeedback({ open: true, message: "Please upload images and select a label of interest", severity: "warning" }); return; }
    try {
      startLoading();
      const results = [];
      // Get label name to pass to prediction (so dummy generator uses readable label)
      const labelObj = labels.find((l) => l.id === selectedLabelId);
      const labelName = labelObj ? labelObj.name : selectedLabelId;
      for (let i = 0; i < files.length; i += 1) {
        // eslint-disable-next-line no-await-in-loop
        const pred = await getPrediction({ input_file: files[i], label_id: labelName, candidate_labels: candidateLabels });
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
      <Typography variant="h5" gutterBottom>Normal Mode</Typography>

      <FileUpload files={files} setFiles={setFiles} />

      <Box sx={{ mt: 2 }}>
        <TextField select label="Label of Interest" value={selectedLabelId} onChange={(e) => setSelectedLabelId(e.target.value)} fullWidth>
          <MenuItem value="">-- select label --</MenuItem>
          {labels.map((l) => (<MenuItem key={l.id} value={l.id}>{l.name}</MenuItem>))}
        </TextField>
      </Box>

      <Box sx={{ display: "flex", gap: 1, mt: 2 }}>
        <TextField label="Candidate Label" value={candidateInput} onChange={(e) => setCandidateInput(e.target.value)} />
        <Button variant="outlined" onClick={handleAddCandidate}>Add</Button>
      </Box>

      <Box sx={{ mt: 1 }}>
        {candidateLabels.map((lbl) => (<Chip key={lbl} label={lbl} onDelete={() => handleRemoveCandidate(lbl)} sx={{ mr: 1, mt: 1 }} />))}
      </Box>

      <Box sx={{ mt: 2 }}>
        <Button variant="contained" onClick={handleSubmit} disabled={!files.length}>Segment & Predict</Button>
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

export default NormalMode;
