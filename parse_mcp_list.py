import requests
from bs4 import BeautifulSoup
import re

def get_awesome_mcp_markdown():
    url = "https://raw.githubusercontent.com/punkpeye/awesome-mcp-servers/main/README.md"
    response = requests.get(url)
    return response.text

def parse_markdown_for_mcps(markdown_text):
    lines = markdown_text.split('\n')
    mcp_entries = []

    pattern = re.compile(r"- \[(.*?)\]\((https?://.*?)\)\s*[-–—]?\s*(.*)")

    for line in lines:
        match = pattern.match(line)
        if match:
            name = match.group(1)
            url = match.group(2)
            desc = match.group(3).strip()
            mcp_entries.append({
                "name": name,
                "url": url,
                "description": desc
            })

    return mcp_entries

if __name__ == "__main__":
    md = get_awesome_mcp_markdown()
    mcps = parse_markdown_for_mcps(md)
    
    for mcp in mcps[:5]:
        print(mcp)
    print(f"\nTotal MCPs parsed: {len(mcps)}")