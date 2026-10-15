from prometheus_client import CollectorRegistry, Counter, Histogram


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "shrimp_http_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "shrimp_http_request_duration_seconds",
            "HTTP latency",
            ["route"],
            registry=self.registry,
        )
        self.labels = Counter(
            "shrimp_predictions_total",
            "Successful aggregated diagnoses",
            ["label"],
            registry=self.registry,
        )
