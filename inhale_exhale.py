import threading


class Cycle:
    """Track size and trigger training when limit exceeded."""

    def __init__(self, threshold: int = 50 * 1024):
        self.threshold = threshold
        self.count = 0
        self.lock = threading.Lock()

    def inhale(self, size: int) -> None:
        """Increase the counter and exhale if the threshold is crossed."""
        with self.lock:
            self.count += size
            if self.count > self.threshold:
                self.exhale()

    def exhale(self) -> None:
        """Start fine-tuning in a separate thread and reset the counter."""
        def target() -> None:
            from le import fine_tune
            fine_tune(async_mode=True)
        threading.Thread(target=target, daemon=True).start()
        with self.lock:
            self.count = 0
