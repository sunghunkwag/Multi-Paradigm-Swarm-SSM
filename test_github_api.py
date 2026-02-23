import urllib.request
import json

try:
    # Test a few variations
    queries = [
        "automl meta-learning -transformer",
        "recursive self improvement",
        "neural architecture search -transformer",
        "program synthesis -llm"
    ]
    
    for q_str in queries:
        print(f"\nTesting query: {q_str}")
        q = urllib.parse.quote(q_str)
        url = f"https://api.github.com/search/repositories?q={q}&sort=stars&order=desc&per_page=3"
        req = urllib.request.Request(url, headers={"User-Agent": "rsi-meta-agent"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            print(f"Total repos found: {data.get('total_count', 0)}")
            for idx, item in enumerate(data.get("items", [])[:3], 1):
                print(f"   {idx}. {item.get('full_name')} (Stars: {item.get('stargazers_count', 0)})")
        except Exception as e:
            print(f"FAILED: {e}")
        
except Exception as e:
    print(f"Error: {type(e).__name__}: {str(e)}")
