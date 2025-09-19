// src/api/training.js
// Model training API calls

import axios from "axios";
import { fetchAPI } from "./index";

// Trigger training for a given model
export const trainModel = async (modelId, params, onProgress) => {
  return axios.post(`/api/v1/${modelId}/train`, params, {
    headers: { "Content-Type": "application/json" },
    onUploadProgress: (event) => {
      if (onProgress) {
        const percent = Math.round((event.loaded * 100) / event.total);
        onProgress(percent);
      }
    },
  });
};

// Fetch training/testing metrics summary
export const getMetricsSummary = () => fetchAPI("/api/v1/metrics/summary");
