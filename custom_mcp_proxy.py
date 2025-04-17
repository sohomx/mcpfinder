from fastapi import FastAPI
from pydantic import BaseModel
import subprocess, json, os, traceback

app = FastAPI()

class RunTopRequest(BaseModel):
    input: dict

@app.post("/run_top")
def run_top(req: RunTopRequest):
    task = req.input.get("task")
    if not task:
        return {"error": "Missing 'task'"}

    binary_path = "/Users/sohom/github-mcp-server/github-mcp-server-binary"
    token = os.getenv("GITHUB_TOKEN")

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "search_repositories",
            "arguments": {
                "query": task
            }
        }
    }

    try:
        print("📤 sending payload:", json.dumps(payload, indent=2))

        proc = subprocess.Popen(
            [binary_path, "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy()
        )

                # send json to stdin with newline to flush properly
        stdout, stderr = proc.communicate(json.dumps(payload) + "\n", timeout=30)

        print("📥 raw stdout:", repr(stdout.strip()))
        print("📥 raw stderr:", repr(stderr.strip()))

        try:
            parsed = json.loads(stdout.strip())
        except Exception as parse_err:
            return {
                "mcp": {
                    "name": "github-mcp-server",
                    "mode": "stdio"
                },
                "proxy_result": {
                    "error": f"Failed to parse JSON from stdout: {str(parse_err)}"
                },
                "debug": {
                    "stdout": stdout.strip(),
                    "stderr": stderr.strip()
                }
            }

        # extract top 3 repos
        items = parsed.get("result", {}).get("content", [])
        try:
            parsed_result = json.loads(items[0]["text"]) if items and "text" in items[0] else {}
            top_repos = parsed_result.get("items", [])[:3]
            simplified = [
                {
                    "name": repo["full_name"],
                    "description": repo.get("description"),
                    "stars": repo.get("stargazers_count"),
                    "url": repo.get("html_url")
                }
                for repo in top_repos
            ]
        except Exception as parse_result_error:
            simplified = {
                "error": "Failed to parse GitHub content",
                "details": str(parse_result_error)
            }

        return {
            "mcp": {
                "name": "github-mcp-server",
                "mode": "stdio"
            },
            "proxy_result": simplified,
            "debug": {
                "stdout": stdout.strip(),
                "stderr": stderr.strip()
            }
        }

    except Exception as e:
        return {
            "mcp": {
                "name": "github-mcp-server",
                "mode": "stdio"
            },
            "proxy_result": {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }