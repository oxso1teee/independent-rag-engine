"""Process metrics in the Prometheus text format, kept in memory and read-only to the outside.

Only counts and sums are exported: request counts by route and status class, request time, answers
by status and the process uptime. Labels are the route *template* (``/v1/documents/{document_id}``,
never a real id), the method and a status class, so a metric never carries a question, a document
name or an identifier. Metrics are off unless ``[api] metrics`` is set, and the endpoint needs the
same token as the rest of the API.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable

_REQUESTS = "rag_http_requests_total"
_SECONDS = "rag_http_request_duration_seconds"
_ANSWERS = "rag_answers_total"


class Metrics:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = defaultdict(int)
        self._seconds: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
        self._answers: dict[str, int] = defaultdict(int)

    def request(self, method: str, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self._requests[(method, route, f"{status // 100}xx")] += 1
            total = self._seconds[(method, route)]
            total[0] += seconds
            total[1] += 1

    def answer(self, status: str) -> None:
        with self._lock:
            self._answers[status] += 1

    def render(self) -> str:
        with self._lock:
            lines = [
                f"# HELP {_REQUESTS} HTTP requests by method, route template and status class.",
                f"# TYPE {_REQUESTS} counter",
            ]
            for (method, route, klass), count in sorted(self._requests.items()):
                lines.append(
                    f'{_REQUESTS}{{method="{method}",route="{route}",status="{klass}"}} {count}'
                )
            lines += [
                f"# HELP {_SECONDS} Time spent serving requests.",
                f"# TYPE {_SECONDS} summary",
            ]
            for (method, route), (total, n) in sorted(self._seconds.items()):
                label = f'method="{method}",route="{route}"'
                lines.append(f"{_SECONDS}_sum{{{label}}} {total:.6f}")
                lines.append(f"{_SECONDS}_count{{{label}}} {int(n)}")
            lines += [
                f"# HELP {_ANSWERS} Answers by status.",
                f"# TYPE {_ANSWERS} counter",
            ]
            for status, count in sorted(self._answers.items()):
                lines.append(f'{_ANSWERS}{{status="{status}"}} {count}')
            uptime = self._clock() - self._started
            lines += [
                "# HELP rag_uptime_seconds Seconds since the server started.",
                "# TYPE rag_uptime_seconds gauge",
                f"rag_uptime_seconds {uptime:.1f}",
            ]
        return "\n".join(lines) + "\n"
