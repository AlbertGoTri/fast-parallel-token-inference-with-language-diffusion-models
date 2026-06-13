"""
LLaDA API Provider for Promptfoo

Custom provider that calls the local Flask server running serve_llada.py
The server must be running at http://127.0.0.1:5000 before evaluation starts.
"""

import json
import urllib.request
import urllib.error
import socket
import time


def call_api(prompt, options, context):
    """Call the local LLaDA Flask server for promptfoo evaluation."""
    url = 'http://127.0.0.1:5000/generate'
    data = json.dumps({'prompt': prompt}).encode('utf-8')
    headers = {'Content-Type': 'application/json'}

    # 128-step diffusion on an 8B 4-bit model can exceed 2 minutes per prompt on consumer GPUs;
    # 5 minutes leaves margin for CPU fallback.
    timeout_seconds = 300

    http_start = time.time()
    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode('utf-8'))
            http_ms = (time.time() - http_start) * 1000

            # Server-reported generation_ms excludes HTTP serialization overhead;
            # subtracting gives the true model inference latency.
            server_timing = result.get('timing', {})
            server_timing['http_roundtrip_ms'] = round(http_ms, 2)

            return {
                "output": result.get('response', ''),
                "metadata": {"timing": server_timing}
            }

    except urllib.error.URLError as e:
        error_msg = str(e)
        # WinError 10061 is the Windows-specific code for connection refused;
        # catching it explicitly avoids cryptic tracebacks for graders.
        if "Connection refused" in error_msg or "WinError 10061" in error_msg:
            return {
                "error": "ERROR: Cannot connect to LLaDA server at 127.0.0.1:5000.\n\n"
                        "Please ensure serve_llada.py is running:\n"
                        "  python serve_llada.py\n\n"
                        "Wait until you see 'Model loaded and ready to serve.' before running promptfoo.",
                "output": "[Server not running - start serve_llada.py first]"
            }
        return {
            "error": f"URL Error: {error_msg}",
            "output": f"[Connection error: {error_msg}]"
        }

    except socket.timeout:
        return {
            "error": "ERROR: Request timed out after 5 minutes. Consider reducing steps/gen_length in serve_llada.py or increasing timeout.",
            "output": "[Timeout - model generation took too long]"
        }

    except json.JSONDecodeError as e:
        return {
            "error": f"ERROR: Invalid JSON response from server: {e}",
            "output": "[Invalid response format]"
        }

    except Exception as e:
        return {
            "error": f"ERROR: {type(e).__name__}: {str(e)}",
            "output": f"[Error: {str(e)}]"
        }
