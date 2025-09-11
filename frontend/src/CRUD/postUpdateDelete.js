// send a post, update or delete request to the API

export async function sendJSON(url, payload = null, method = 'POST', options = {}) {
  const controller = new AbortController();
  const { signal } = controller;
  const timeout = options.timeout || 20000;
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const res = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        ...(options.headers || {}),
      },
      body: payload != null ? JSON.stringify(payload) : undefined,
      signal,
    });

    clearTimeout(timeoutId);

    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch (e) { data = text; }

    if (!res.ok) {
      const message = (data && data.message) || `Request failed: ${res.status}`;
      throw new Error(message);
    }

    return { data, status: res.status };
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('Request aborted or timed out');
    throw err;
  }
}
