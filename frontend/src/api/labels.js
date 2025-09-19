import { dummyLabels } from "../data/dummy";

// Helper to decide whether to use real API or dummy
const useDummy = true; // toggle for development

export const getLabels = async () => {
  if (useDummy) {
    return new Promise((res) => setTimeout(() => res(dummyLabels), 500));
  }
  const response = await fetch("/api/v1/labels");
  return response.json();
};

export const createLabel = async (data) => {
  if (useDummy) return new Promise((res) => setTimeout(res, 300));
  const response = await fetch("/api/v1/labels", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return response.json();
};

export const updateLabel = async (id, data) => {
  if (useDummy) return new Promise((res) => setTimeout(res, 300));
  const response = await fetch(`/api/v1/labels/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return response.json();
};

export const deleteLabel = async (id) => {
  if (useDummy) return new Promise((res) => setTimeout(res, 300));
  const response = await fetch(`/api/v1/labels/${id}`, { method: "DELETE" });
  return response.json();
};
