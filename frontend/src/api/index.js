// src/api/index.js
// Central fetch wrapper with error handling

export const fetchAPI = async (url, options = {}) => {
  try {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) throw new Error(`Error ${res.status}: ${res.statusText}`);
    return await res.json();
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
};
