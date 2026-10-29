class ServiceError(Exception):
    def __init__(self, code: str, message: str, status=422, headers=None):
        self.code = code
        self.message = message
        self.status = status
        self.headers = headers or {}
        super().__init__(message)
