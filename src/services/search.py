"""
Tool for search, expand information for question form user"
"""

import requests
from typing import List, Dict, Optional

def brave_search(query: str, api_key: str, count: int = 5) -> Dict:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json","X-Subscription-Token": api_key}
    params = {"q": query,"count": count}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    
    data = response.json()
    results = []
    for item in data.get("web", {}).get("results", []):
        results.append({"title": item.get("title"),"url": item.get("url"),"description": item.get("description")})
    return { "query": query, "results": results,"total": len(results)}


if __name__ == "__main__":
    API_KEY = "API"
    result = brave_search("Triệu chứng của chest x-ray pneumonia", API_KEY)
    
    print(f"Query: {result['query']}")
    print(f"Total: {result['total']}\n")
    
    for i, r in enumerate(result['results'], 1):
        print(f"{i}. {r['title']}")
        print(f"   {r['url']}\n")
