"""
LLaDA API Provider for Promptfoo

Custom provider that calls the local Flask server running serve_llada.py
The server must be running at http://127.0.0.1:5000 before evaluation starts.
"""

import json
import urllib.request
import urllib.error
import socket


def call_api(prompt, options, context):
    """
    Custom promptfoo provider for LLaDA model.
    Calls the local Flask server running serve_llada.py
    """
    url = 'http://127.0.0.1:5000/generate'
    data = json.dumps({'prompt': prompt}).encode('utf-8')
    headers = {'Content-Type': 'application/json'}

    # Use longer timeout for LLaDA generation (can take several minutes)
    timeout_seconds = 300  # 5 minutes

    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode('utf-8'))
            return {"output": result.get('response', '')}

    except urllib.error.URLError as e:
        error_msg = str(e)
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
