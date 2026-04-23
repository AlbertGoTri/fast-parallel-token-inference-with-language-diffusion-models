/**
 * ollama_judge_provider.js
 *
 * Custom promptfoo provider - Uses local Ollama model (llama3.1:8b) as judge.
 *
 * No rate limiting needed since this runs locally. Each evaluation call
 * queries Ollama at http://127.0.0.1:11434 (default Ollama port).
 *
 * Requirements:
 *   - Ollama must be installed and running
 *   - Model llama3.1:8b must be pulled: ollama pull llama3.1:8b
 */

// ─── Serial queue (module-level singleton) ────────────────────────────────────
// Queue ensures evaluations run sequentially to avoid VRAM collision with LLaDA

let _queue = Promise.resolve();
let _queueLength = 0;

function _enqueue(task) {
  _queueLength++;
  const result = _queue.then(async () => {
    try {
      return await task();
    } finally {
      _queueLength--;
    }
  });
  _queue = result.catch(() => {});
  return result;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function _post(url, payload) {
  return import('http').then((mod) => {
    const http = mod.default ?? mod;
    return new Promise((resolve, reject) => {
      const data   = JSON.stringify(payload);
      const urlObj = new URL(url);
      const req    = http.request(
        {
          hostname: urlObj.hostname,
          port:     urlObj.port || 11434,
          path:     urlObj.pathname + urlObj.search,
          method:   'POST',
          headers: {
            'Content-Type':   'application/json',
            'Content-Length': Buffer.byteLength(data),
          },
        },
        (res) => {
          let raw = '';
          res.on('data', (c) => (raw += c));
          res.on('end', () => {
            if (res.statusCode >= 200 && res.statusCode < 300) {
              try { resolve(JSON.parse(raw)); }
              catch (e) { reject({ status: 0, body: `JSON parse error: ${e.message}` }); }
            } else {
              reject({ status: res.statusCode, body: raw });
            }
          });
        }
      );
      req.on('error', (e) => reject({ status: 0, body: e.message }));
      req.setTimeout(300_000, () => {  // 5 minute timeout for local model
        req.destroy();
        reject({ status: 0, body: 'Request timeout (300 s)' });
      });
      req.write(data);
      req.end();
    });
  });
}

// ─── Core Ollama call ─────────────────────────────────────────────────────────

async function _callOllama(prompt) {
  const url = 'http://127.0.0.1:11434/api/generate';

  const payload = {
    model: 'llama3.1:8b',
    prompt: prompt,
    stream: false,
    options: {
      temperature: 0,
      num_predict: 256
    }
  };

  try {
    const data = await _post(url, payload);
    const text = data.response;

    if (!text) {
      return {
        error: 'Empty response from Ollama',
        pass: false,
        score: 0,
        reason: 'No response text from Ollama'
      };
    }

    // Parse the JSON response to extract Yes/No
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      // Try to extract JSON from markdown if present
      const jsonMatch = text.match(/\{[\s\S]*?\}/);
      if (jsonMatch) {
        try {
          parsed = JSON.parse(jsonMatch[0]);
        } catch (e2) {
          parsed = null;
        }
      }
    }

    // Return standardized format
    if (parsed && parsed.answer) {
      const isPass = parsed.answer.toLowerCase() === 'yes';
      return {
        pass: isPass,
        score: isPass ? 1 : 0,
        reason: parsed.reason || text,
        output: text
      };
    }

    // Fallback: check text for Yes/No
    const textLower = text.toLowerCase();
    const hasYes = textLower.includes('"yes"') || textLower.includes("yes") || textLower.includes("yes");
    const hasNo = textLower.includes('"no"') || textLower.includes("no") || textLower.includes("no");

    if (hasYes && !hasNo) {
      return { pass: true, score: 1, reason: text, output: text };
    } else if (hasNo && !hasYes) {
      return { pass: false, score: 0, reason: text, output: text };
    }

    // If unclear, default to parsing as pass if no explicit "no"
    return { pass: true, score: 1, reason: text, output: text };

  } catch (err) {
    return {
      error: `Ollama error: ${err.body || err.message}`,
      pass: false,
      score: 0,
      reason: `API Error: ${err.body || err.message}`
    };
  }
}

// ─── Provider class ───────────────────────────────────────────────────────────

export default class OllamaJudgeProvider {
  id() {
    return 'ollama-llama3.1-8b-judge';
  }

  async callApi(prompt, options, context) {
    // Check if Ollama is available
    try {
      await import('http').then((mod) => {
        const http = mod.default ?? mod;
        return new Promise((resolve, reject) => {
          const req = http.request(
            { hostname: '127.0.0.1', port: 11434, path: '/', method: 'GET', timeout: 5000 },
            (res) => resolve(res.statusCode),
            (e) => reject(e)
          );
          req.on('error', reject);
          req.end();
        });
      });
    } catch (e) {
      return {
        error: 'Cannot connect to Ollama at 127.0.0.1:11434. Is Ollama running?',
        pass: false,
        score: 0,
        reason: 'Ollama not running'
      };
    }

    return _enqueue(() => _callOllama(prompt));
  }
}
