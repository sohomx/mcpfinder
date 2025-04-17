import requests

def invoke_mcp_via_proxy(mcp_url, task_input):
    proxy_url = "https://proxy.pluggedin.dev/run"

    payload = {
        "url": mcp_url,
        "input": {
            "task": task_input
        }
    }

    try:
        response = requests.post(proxy_url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Failed to invoke MCP: {str(e)}"}
    
def extract_slug_for_pluggedin(name, url):
    try:
        from urllib.parse import urlparse
        parts = urlparse(url).path.strip("/").split("/")
        if len(parts) >= 2:
            owner = parts[0]
            repo = parts[1].replace("-", "_")
            return f"{owner}__{repo}".lower()
    except Exception as e:
        print(f"❌ failed to extract slug for {name}: {e}")
    return None

def extract_slug_for_pluggedin(_, url):
    # assumes url like: https://github.com/anaisbetts/mcp-youtube
    try:
        slug = url.split("github.com/")[-1]  # -> anaisbetts/mcp-youtube
        return slug.replace("/", "__")       # -> anaisbetts__mcp-youtube
    except Exception:
        return None

def extract_slug_for_pluggedin(_, url):
    try:
        slug = url.split("github.com/")[-1]
        return slug.replace("/", "__")
    except Exception:
        return None

def invoke_mcp_via_proxy(url, task):
    slug = extract_slug_for_pluggedin("unused", url)
    if not slug:
        return {"error": "Invalid MCP URL"}

    try:
        response = requests.post(
            "http://localhost:3030/run",  # ✅ local proxy, NOT pluggedin.dev
            headers={"Content-Type": "application/json"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": slug,
                    "arguments": {
                        "task": task  # ✅ match pluggedin tool input
                    }
                }
            },
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": f"Failed to invoke MCP: {e}"}