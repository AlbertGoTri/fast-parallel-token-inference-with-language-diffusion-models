/**
 * gemini_judge_provider.js
 *
 * Custom promptfoo provider - Gemini 2.5 Flash judge with strict rate limiting.
 *
 * RATE LIMITING STRATEGY:
 *   The free tier officially allows 15 RPM, but in practice:
 *   - Google AI Studio keys get rate limited much sooner (~60 requests/hour)
 *   - Each project/user has additional daily/hourly quotas
 *   - This provider uses ultra-conservative settings:
 *     - 2 RPM (one token every 30 seconds)
 *     - Minimum 30 second gap between calls
 *     - After a 429, wait 2 minutes + 1 minute per consecutive error
 *
 *   With 60 assertions at 2 RPM = ~30 minutes total runtime.
 *   Set PROMPTFOO_REQUEST_TIMEOUT_MS=7200000 (2 hours) to accommodate this.
 */

// ─── Rate limiting (module-level singletons) ──────────────────────────────────

const RPM          = 2;                   // Ultra-conservative: 2 requests per minute
const RATE_PER_MS  = RPM / 60_000;
const BUCKET_MAX   = 1;                   // Only allow 1 request at a time
const MIN_GAP_MS   = 30_000;             // Hard minimum 30 seconds between calls

let _tokens      = 1;                     // Start with 1 token
let _lastRefill  = Date.now();
let _lastCallAt  = 0;
let _consecutive429s = 0;               // Track consecutive rate limit errors

function _refill() {
  const now   = Date.now();
  const delta = now - _lastRefill;
  _tokens     = Math.min(BUCKET_MAX, _tokens + delta * RATE_PER_MS);
  _lastRefill = now;
}

// ─── Serial queue (module-level singleton) ────────────────────────────────────

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

async function _acquireSlot() {
  while (true) {
    _refill();
    const now           = Date.now();
    const sinceLastCall = now - _lastCallAt;
    const gapOk         = sinceLastCall >= MIN_GAP_MS;
    const tokenOk       = _tokens >= 1;

    if (tokenOk && gapOk) {
      _tokens     -= 1;
      _lastCallAt  = now;
      return;
    }

    // Calculate wait time for whichever constraint is blocking
    const waitForToken = tokenOk ? 0 : Math.ceil((1 - _tokens) / RATE_PER_MS);
    const waitForGap   = gapOk ? 0 : Math.ceil(MIN_GAP_MS - sinceLastCall);
    const waitMs       = Math.max(waitForToken, waitForGap, 1000);

    console.log(`[gemini_judge] Rate limiting: waiting ${waitMs}ms (token=${_tokens.toFixed(2)}, gap=${sinceLastCall}ms, queue=${_queueLength})`);
    await sleep(waitMs);
  }
}

function _post(url, payload) {
  return import('https').then((mod) => {
    const https = mod.default ?? mod;
    return new Promise((resolve, reject) => {
      const data   = JSON.stringify(payload);
      const urlObj = new URL(url);
      const req    = https.request(
        {
          hostname: urlObj.hostname,
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
      req.setTimeout(120_000, () => {
        req.destroy();
        reject({ status: 0, body: 'Request timeout (120 s)' });
      });
      req.write(data);
      req.end();
    });
  });
}

// ─── Core Gemini call ─────────────────────────────────────────────────────────

async function _callGemini(prompt, apiKey) {
  const url =
    'https://generativelanguage.googleapis.com/v1beta/models/' +
    `gemini-2.5-flash:generateContent?key=${apiKey}`;

  const payload = {
    contents:           [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig:   { temperature: 0, maxOutputTokens: 256 },
  };

  let attempt = 0;
  const maxAttempts = 20;

  while (attempt < maxAttempts) {
    attempt++;
    await _acquireSlot();

    try {
      const data = await _post(url, payload);
      const text = data.candidates[0].content.parts[0].text;

      // Reset consecutive error counter on success
      _consecutive429s = 0;

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
      const hasYes = textLower.includes('"yes"') || textLower.includes("answer\": \"yes") || textLower.includes('answer": "yes');
      const hasNo = textLower.includes('"no"') || textLower.includes("answer\": \"no") || textLower.includes('answer": "no');

      if (hasYes && !hasNo) {
        return { pass: true, score: 1, reason: text, output: text };
      } else if (hasNo && !hasYes) {
        return { pass: false, score: 0, reason: text, output: text };
      }

      // If unclear, default to parsing as pass if no explicit "no"
      return { pass: true, score: 1, reason: text, output: text };

    } catch (err) {
      if (err.status === 429) {
        _consecutive429s++;
        // Much more aggressive backoff for 429s
        // FREE TIER LIMIT IS ~60 reqs/hour (~1/minute), so after hitting 429, wait 3-5 minutes
        const baseWait = 180_000; // 3 minutes base wait
        const waitMs = baseWait + (_consecutive429s * 60_000); // Add 1 min per consecutive error

        // Cap at 10 minutes max wait
        const finalWait = Math.min(waitMs, 600_000);

        console.warn(
          `[gemini_judge] 429 on attempt ${attempt} (${_consecutive429s} consecutive) — ` +
          `waiting ${Math.round(finalWait/1000)}s. ` +
          `Consider using a paid API tier or running fewer tests.`
        );

        // Reset rate limiter state completely - force full cooldown
        _tokens = 0;
        _lastCallAt = Date.now() + finalWait; // Pretend we just made a call, force full wait
        await sleep(finalWait);
        continue;
      }

      // Non-429 error: return it
      return {
        error: `HTTP ${err.status}: ${err.body}`,
        pass: false,
        score: 0,
        reason: `API Error: ${err.body}`
      };
    }
  }

  // Max attempts reached
  return {
    error: `Max attempts (${maxAttempts}) reached`,
    pass: false,
    score: 0,
    reason: 'Rate limit: too many retries'
  };
}

// ─── Provider class ───────────────────────────────────────────────────────────

export default class GeminiJudgeProvider {
  id() {
    return 'gemini-2.5-flash-judge';
  }

  async callApi(prompt, options, context) {
    const apiKey = process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return {
        error: 'No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY.',
        pass: false,
        score: 0
      };
    }
    return _enqueue(() => _callGemini(prompt, apiKey));
  }
}
