// send a get request for to the API
export async function getJSON(url, options = {}) {
  const controller = new AbortController();
  const { signal } = controller;
  const fetchOptions = { method: 'GET', signal, headers: { 'Accept': 'application/json', ...(options.headers || {}) } };

  // caller can pass a timeout in ms
  const timeout = options.timeout || 15000;
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const res = await fetch(url, fetchOptions);
    clearTimeout(timeoutId);
    if (!res.ok) {
      const text = await res.text();
      let errMsg = `Request failed: ${res.status}`;
      try { errMsg = JSON.parse(text).message || errMsg; } catch (e) {}
      throw new Error(errMsg);
    }
    const json = await res.json();
    return { data: json, status: res.status };
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Request aborted or timed out');
    throw err;
  }
}
