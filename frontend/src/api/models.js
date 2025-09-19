// src/api/models.js
import { dummyModels as initialDummyModels } from "../data/dummy";

const USE_DUMMY = true;

// In-memory store for development
let _modelsStore = initialDummyModels.map((m) => ({ ...m }));

// simulated delay
const delay = (ms = 300) => new Promise((res) => setTimeout(res, ms));

export const getModels = async () => {
  if (USE_DUMMY) {
    await delay(200);
    // return a clone
    return _modelsStore.map((m) => ({ ...m }));
  }

  const res = await fetch("/api/v1/models");
  return res.json();
};

export const createModel = async (data) => {
  if (USE_DUMMY) {
    await delay(200);
    const id = `m${Date.now()}`;
    const model = { id, ...data };
    _modelsStore.push(model);
    return { ...model };
  }

  const res = await fetch("/api/v1/models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
};

export const updateModel = async (id, data) => {
  if (USE_DUMMY) {
    await delay(200);
    _modelsStore = _modelsStore.map((m) => (m.id === id ? { ...m, ...data } : m));
    return _modelsStore.find((m) => m.id === id);
  }

  const res = await fetch(`/api/v1/models/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
};

export const deleteModel = async (id) => {
  if (USE_DUMMY) {
    await delay(200);
    _modelsStore = _modelsStore.filter((m) => m.id !== id);
    return { ok: true };
  }

  const res = await fetch(`/api/v1/models/${id}`, { method: "DELETE" });
  return res.json();
};
