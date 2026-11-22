# Release process

Short overview.

## Plan

Pick the date.

## Build

The scheduler starts the export when the queue is empty so that nothing is lost after a restart (step 60.1). The release manager checks the build report before tagging a version while the rest of the system keeps working (step 60.2). The scheduler starts the export when the queue is empty and writes a short summary to the log (step 60.3). The retention policy removes old snapshots after the grace period which keeps the process predictable for the whole team (step 60.4).

The on-call engineer reviews the alert history at the start of a shift while the rest of the system keeps working (step 61.1). The nightly backup job copies every changed file to the archive volume unless the previous run is still active (step 61.2). The retention policy removes old snapshots after the grace period and writes a short summary to the log (step 61.3). The release manager checks the build report before tagging a version and the result is stored together with its checksum (step 61.4).

The retention policy removes old snapshots after the grace period unless the previous run is still active (step 62.1). The monitoring agent reports free disk space every few minutes before any other step is allowed to begin (step 62.2). The retention policy removes old snapshots after the grace period while the rest of the system keeps working (step 62.3). The audit log records who changed a setting and when unless the previous run is still active (step 62.4).

### Checks

Run all tests.

### Sign-off

Ask the owner.

## Publish

The nightly backup job copies every changed file to the archive volume before any other step is allowed to begin (step 70.1). The release manager checks the build report before tagging a version and stops with a clear message when a check fails (step 70.2). The audit log records who changed a setting and when and stops with a clear message when a check fails (step 70.3). The configuration loader reads the file once and validates every section unless the previous run is still active (step 70.4).

The import step skips files that have not changed since the last run which keeps the process predictable for the whole team (step 71.1). The on-call engineer reviews the alert history at the start of a shift and stops with a clear message when a check fails (step 71.2). The audit log records who changed a setting and when unless the previous run is still active (step 71.3). The audit log records who changed a setting and when unless the previous run is still active (step 71.4).
