// src/api/outputs.js
// Simple client-side API wrapper; falls back to dummy generation if no real API available.

import { generateFullDummyPrediction } from "../data/dummy";

const USE_DUMMY = true; // toggle to hit real endpoints when ready

/**
 * getPrediction
 * Params: { input_file: File, label_id: string (label of interest), candidate_labels: string[] }
 * Returns: prediction object matching generateFullDummyPrediction structure
 */
export const getPrediction = async ({ input_file, label_id, candidate_labels = [] } = {}) => {
  if (USE_DUMMY) {
    // If input_file is a File from FileUpload, pass it
    return new Promise((res) => {
      setTimeout(() => {
        res(generateFullDummyPrediction({
          file: input_file,
          labelOfInterest: label_id || "LabelOfInterest",
          candidateLabels: candidate_labels || [],
        }));
      }, 300); // small delay
    });
  }

  // If you'd like to call your real API, implement fetch/XHR here.
  const response = await fetch("/api/v1/outputs", {
    method: "POST",
    body: JSON.stringify({ input_file: input_file?.name, label_id, candidate_labels }),
    headers: { "Content-Type": "application/json" },
  });
  return response.json();
};

export const saveCorrectedCount = async ({ output_id, label, corrected_count } = {}) => {
  if (USE_DUMMY) {
    // simulate success
    return new Promise((res) => setTimeout(() => res({ ok: true }), 200));
  }
  const response = await fetch(`/api/v1/outputs/${output_id}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, corrected_count }),
  });
  return response.json();
};
