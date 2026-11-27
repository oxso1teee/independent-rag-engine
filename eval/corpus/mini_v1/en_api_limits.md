# API rate limits

Each API key may send 100 requests per minute, with a burst allowance of 20 extra requests. When the limit is exceeded the service answers with HTTP status 429 and a Retry-After header that tells the client how many seconds to wait. Enterprise plans raise the limit to 1000 requests per minute.
