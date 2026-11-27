# Key management

Cryptographic keys are rotated every 90 days. Administrators sign in with hardware security tokens, and secrets live in the central secrets manager. Keys must never be committed to a repository; a scanner blocks pushes that contain them.
