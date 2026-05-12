import threading

class ApiResult:
    def __init__(self):
        self.event = threading.Event()
        self.value = None
        self.error = None

    def set_result(self, value):
        self.value = value
        self.event.set()
    
    def set_error(self, error):
        self.error = error
        self.event.set()

    def result(self, timeout=None):
        finished = self.event.wait(timeout)

        if not finished:
            return None

        if self.error is not None:
            return None

        return self.value