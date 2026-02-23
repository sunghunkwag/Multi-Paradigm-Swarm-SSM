"""
Jules API Client for RSI System
Google Jules Coding Agent - REST API Integration
"""
import os
import json
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List


class JulesAPIClient:
    """Client for interacting with Google Jules Coding Agent API."""
    
    BASE_URL = "https://jules.googleapis.com/v1alpha"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._load_api_key()
        self.headers = {
            "X-Goog-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        self.sources: List[Dict] = []
        
    def _load_api_key(self) -> str:
        """Load API key from file."""
        key_paths = [
            r"c:\Users\starg\OneDrive\바탕 화면\SCIG-RSI-v2\google_api_key.txt",
            "google_api_key.txt",
            "../google_api_key.txt",
        ]
        for path in key_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        key = f.read().strip().split('\n')[0].strip()
                        if key and len(key) > 20:
                            print(f"[Jules] Loaded API key from {path}")
                            return key
            except Exception:
                pass
        raise ValueError("Jules API key not found")
    
    def list_sources(self) -> List[Dict]:
        """List available GitHub sources connected to Jules."""
        url = f"{self.BASE_URL}/sources"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            self.sources = data.get("sources", [])
            print(f"[Jules] Found {len(self.sources)} connected sources")
            return self.sources
        except requests.exceptions.RequestException as e:
            print(f"[Jules] Error listing sources: {e}")
            return []
    
    def create_session(self, source_id: str, prompt: str) -> Optional[Dict]:
        """
        Create a new session with Jules.
        
        Args:
            source_id: The source (repository) ID to work on
            prompt: The task description for Jules
            
        Returns:
            Session data including session_id
        """
        url = f"{self.BASE_URL}/sessions"
        payload = {
            "source": source_id,
            "prompt": prompt
        }
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            session = response.json()
            print(f"[Jules] Created session: {session.get('name', 'unknown')}")
            return session
        except requests.exceptions.RequestException as e:
            print(f"[Jules] Error creating session: {e}")
            return None
    
    def list_activities(self, session_id: str) -> List[Dict]:
        """List activities in a session."""
        url = f"{self.BASE_URL}/sessions/{session_id}/activities"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("activities", [])
        except requests.exceptions.RequestException as e:
            print(f"[Jules] Error listing activities: {e}")
            return []
    
    def approve_plan(self, session_id: str) -> bool:
        """Approve Jules' plan to proceed with execution."""
        url = f"{self.BASE_URL}/sessions/{session_id}:approvePlan"
        try:
            response = requests.post(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            print("[Jules] Plan approved!")
            return True
        except requests.exceptions.RequestException as e:
            print(f"[Jules] Error approving plan: {e}")
            return False
    
    def wait_for_completion(self, session_id: str, timeout: int = 300) -> Dict:
        """
        Wait for Jules to complete the task.
        
        Args:
            session_id: The session ID to monitor
            timeout: Maximum wait time in seconds
            
        Returns:
            Final session state with results
        """
        start_time = time.time()
        last_status = ""
        
        while time.time() - start_time < timeout:
            activities = self.list_activities(session_id)
            
            for activity in activities:
                activity_type = activity.get("activityType", "")
                status = activity.get("status", "")
                
                if status != last_status:
                    print(f"[Jules] Status: {activity_type} - {status}")
                    last_status = status
                
                # Check for completion
                if activity_type == "PLAN_GENERATED" and status == "PENDING_APPROVAL":
                    print("[Jules] Plan generated, auto-approving...")
                    self.approve_plan(session_id)
                    
                if activity_type == "EXECUTION_COMPLETE":
                    return {
                        "status": "complete",
                        "activities": activities,
                        "result": activity.get("result", {})
                    }
                    
                if status == "FAILED":
                    return {
                        "status": "failed",
                        "activities": activities,
                        "error": activity.get("error", "Unknown error")
                    }
            
            time.sleep(5)  # Poll every 5 seconds
        
        return {"status": "timeout", "activities": activities}
    
    def improve_code(self, source_id: str, file_path: str, function_name: str, 
                     improvement_hints: str = "") -> Optional[str]:
        """
        Request Jules to improve a specific function.
        
        Args:
            source_id: Repository source ID
            file_path: Path to file in repository
            function_name: Name of function to improve
            improvement_hints: Optional hints about what to improve
            
        Returns:
            Improved code or None if failed
        """
        prompt = f"""Improve the function `{function_name}` in file `{file_path}`.

Focus on:
1. Performance optimization
2. Better algorithms
3. Code clarity
4. Error handling

{improvement_hints}

Create a pull request with the improvements.
"""
        session = self.create_session(source_id, prompt)
        if not session:
            return None
        
        session_id = session.get("name", "").split("/")[-1]
        result = self.wait_for_completion(session_id)
        
        if result.get("status") == "complete":
            return result.get("result", {}).get("pull_request_url")
        
        return None


class JulesRSIIntegration:
    """Integration of Jules API with RSI System."""
    
    def __init__(self, api_key: Optional[str] = None):
        try:
            self.client = JulesAPIClient(api_key)
            self.available = True
            self.sources = self.client.list_sources()
        except Exception as e:
            print(f"[Jules] Initialization failed: {e}")
            self.available = False
            self.sources = []
    
    def find_source_by_name(self, repo_name: str) -> Optional[str]:
        """Find source ID by repository name."""
        for source in self.sources:
            if repo_name.lower() in source.get("name", "").lower():
                return source.get("name")
        return None
    
    def request_improvement(self, repo_name: str, file_path: str, 
                           function_name: str, context: Dict[str, Any] = None) -> Dict:
        """
        Request Jules to improve a function in the RSI codebase.
        
        Args:
            repo_name: Name of the GitHub repository
            file_path: Path to file in the repository
            function_name: Function to improve
            context: Current performance metrics and hints
            
        Returns:
            Result dict with status and any PR URL
        """
        if not self.available:
            return {"status": "unavailable", "error": "Jules API not available"}
        
        source_id = self.find_source_by_name(repo_name)
        if not source_id:
            return {"status": "error", "error": f"Repository {repo_name} not connected to Jules"}
        
        # Build improvement hints from context
        hints = ""
        if context:
            hints = f"""
Current performance metrics:
- improvement_ema: {context.get('improvement_ema', 'N/A')}
- best_scalar_ema: {context.get('best_scalar_ema', 'N/A')}
- pareto_size: {context.get('pareto_size', 'N/A')}

The system is trying to achieve recursive self-improvement.
Focus on optimizations that will improve these metrics.
"""
        
        pr_url = self.client.improve_code(source_id, file_path, function_name, hints)
        
        if pr_url:
            return {"status": "success", "pull_request": pr_url}
        else:
            return {"status": "failed", "error": "Jules could not complete the improvement"}


def test_jules_api():
    """Test the Jules API integration."""
    print("=" * 60)
    print("Jules API Integration Test")
    print("=" * 60)
    
    try:
        integration = JulesRSIIntegration()
        
        if not integration.available:
            print("[FAIL] Jules API not available")
            return False
        
        print(f"[PASS] Jules API available, {len(integration.sources)} sources connected")
        
        # List sources
        for source in integration.sources:
            print(f"  - {source.get('name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


if __name__ == "__main__":
    test_jules_api()
