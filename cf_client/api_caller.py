import requests
import threading
import queue
import time
from .api_result import ApiResult

class ApiCaller:
    BASE_URL = "https://codeforces.com/api"
    DEFAULT_INTERVAL_SEC = 2.0

    def __init__(self):
        self.interval_sec = self.DEFAULT_INTERVAL_SEC
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.worker_thread.start()

    def stop(self):
        self.queue.put(None)
        self.worker_thread.join()

    def enqueue(self, method, params):
        result = ApiResult()
        self.queue.put((method, params, result))
        return result

    def run(self):
        while True:
            task = self.queue.get()
            
            try:
                if task is None:
                    break
                
                method, params, result = task
                value = self.call_api(method, params)
                result.set_result(value)
            except Exception as e:
                result.set_error(e)
            finally:
                self.queue.task_done()
            
            time.sleep(self.interval_sec)

    def call_api(self, method, params):
        url = f"{self.BASE_URL}/{method}"
        
        try: 
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()
        except requests.Timeout as e:
            raise TimeoutError(f"Codeforces API timeout: {method}") from e
        except requests.RequestException as e:
            raise RuntimeError(f"Codeforces API request failed: {method}") from e
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON from Codeforces API: {method}") from e

        if data.get("status") != "OK":
            comment = data.get("comment", "unknown error")
            raise RuntimeError(f"Codeforces API failed: {comment}")

        if "result" not in data:
            raise RuntimeError("Codeforces API response missing result")
        
        return data["result"]
