# Backup policy

All production databases are backed up every night at 02:30 UTC. Daily copies are kept for 14 days and weekly copies for 8 weeks. Every copy is encrypted with AES-256 before it leaves the data centre. A restore test is run on the first Monday of each month, and the result is recorded in the operations log.
