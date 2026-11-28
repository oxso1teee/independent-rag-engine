# Security policy

## Reporting

Report a vulnerability privately to the repository owner through GitHub's private vulnerability
reporting (Security tab), or by email to the address on the owner's GitHub profile. Do not open a
public issue for it. Please include the version, a reproduction and the impact. There is no bounty.

## Scope and threat model

The engine is a local, single-user service. It assumes an honest operating system account and a
trusted machine; it defends against mistakes, hostile *documents* and hostile *network neighbours*.

| Threat | Defence |
|---|---|
| Another host reaches the API | binds `127.0.0.1` by default; any other bind refuses to start without a bearer token of at least 16 characters; the token is compared in constant time and never logged or echoed |
| The API is used to read or write arbitrary files | the API registers no paths; it writes only into engine-managed sources, with clean relative keys, a supported extension, atomic writes and a size limit; responses carry no server paths |
| A browser page calls the local API | CORS is off unless origins are listed |
| Oversized or malformed requests | body, question and page-size limits; strict request schemas; problem+json errors that reveal nothing about internals |
| A document tries to instruct the model or forge a citation | passages are data in the prompt, marker-like text inside them is neutralised, and a citation is accepted only if it names evidence of that request; text with failed citations is withheld |
| Filter or query injection | filters are a typed DSL compiled to parameterised SQL; user text reaches FTS only as quoted terms |
| A malicious file crashes or exhausts the parser | parsers run in an isolated worker with size, time and memory limits; a failure is a typed error, not a crash |
| Secrets in files or logs | secrets are read from environment variables only (the configuration holds the variable *name*); logs never contain document text, prompts, answers or secrets |
| A tampered model file | model files are pinned by a directory fingerprint and verified at load; nothing is downloaded at run time |
| A vulnerable dependency | `pip-audit` and a licence check run in CI; the lock file is checked and the Docker image installs from hashes |

Not defended: a compromised account on the same machine, a malicious operator, denial of service by
a client that holds the token, TLS (put a reverse proxy in front for a non-loopback bind), and
side channels. The idempotency memory and the metrics live in the server process.

## Supported versions

Only the latest commit of the default branch. Pre-release: the API under `/v1` is stable within a
minor series once a release is cut, and no release has been cut.
