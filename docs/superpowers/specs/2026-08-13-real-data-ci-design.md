# Real-data-only CI design

## Goal

Every GitHub Actions run that executes the fraud pipeline or publishes evidence
must use the genuine ULB Credit Card Fraud dataset from OpenML dataset 1597. If
the real dataset cannot be fetched or verified, the job must fail. It must never
fall back to synthetic data.

Synthetic data remains available for local development and unit tests.

## Current behavior and root cause

The repository currently has three synthetic CI paths:

- The regular CI workflow runs `make verify` in a clean checkout. Its `data`
  prerequisite generates synthetic data when `data/raw/creditcard.csv` is
  missing.
- The Docker verification image excludes raw data and runs `make verify`, which
  generates synthetic data inside the container.
- The DVC clean-clone verifier explicitly runs `python -m src.simulate_data`.

Scheduled monitoring attempts the real OpenML download, but marks that step
`continue-on-error: true`. A failed download therefore reaches `make verify`,
which silently generates synthetic data and continues to publish evidence.

## Selected approach

Use an explicit real-data contract at every GitHub Actions pipeline boundary.
Keep the existing local synthetic path instead of adding implicit behavior
based on GitHub's `CI` environment variable.

This makes each CI workflow readable: it fetches real data, stops if the fetch
fails, and only then runs pipeline verification. It also avoids changing the
convenient offline behavior of ordinary local commands.

## Workflow behavior

### Regular CI

The verification job will fetch the real ULB dataset before any pipeline or DVC
reproduction step. The fetch step will not allow failure. Complete pipeline
verification and uploaded evidence will therefore be generated from real data.

Fast unit tests may still exercise the synthetic generator in isolated test
directories; they do not generate pipeline evidence and are outside the
real-data restriction.

### DVC clean-clone reproduction

The clean-clone verifier will support an explicit real-data mode used by CI. In
that mode it will copy the already-fetched raw CSV and its provenance record
into the disposable clone, fail if either is unavailable or not real, and run
the DVC graph from those inputs.

The existing synthetic mode will remain the default for local clean-clone
development so offline contributors can still exercise the DVC graph without
downloading the full dataset.

### Docker verification

The CI Docker run will explicitly fetch the real dataset inside the container
before invoking the shared verification command. A failed download will stop
the container and fail the job. The image can retain its current local default
command because only the GitHub Actions execution contract is being changed.

### Scheduled monitoring

The OpenML fetch will no longer use `continue-on-error`. The monitoring pipeline,
dashboard upload, and Pages deployment will run only after a successful real
data fetch. The run summary will state that the dataset is the real ULB dataset;
the synthetic-fallback branch will be removed.

## Failure handling

- OpenML download, checksum, conversion, provenance, or real-data validation
  failure stops the relevant job.
- No GitHub Actions step catches the failure and invokes
  `src.simulate_data`.
- Existing resumable-download behavior remains unchanged, so interrupted
  transfers retain their `.part` file for the duration of the runner.
- Evidence and Pages artifacts are not uploaded or deployed after a failed
  real-data prerequisite.

## Verification strategy

Regression tests will parse the workflow YAML and inspect the DVC verifier to
enforce these invariants:

1. Regular CI fetches real data before DVC and complete verification.
2. Scheduled monitoring does not allow the fetch step to fail and contains no
   synthetic-fallback summary branch.
3. Docker CI explicitly runs the real-data fetch before verification.
4. CI's DVC command enables real-data mode.
5. Real-data DVC mode rejects missing, synthetic, or invalid provenance rather
   than generating replacement data.
6. Local synthetic generation tests remain valid.

The regression tests will be written and observed failing before workflow or
script implementation changes. After the minimal changes pass those tests, the
full relevant test suite and dry-run Make contracts will be executed.

## Documentation

References that currently describe synthetic data as a CI fallback will be
updated to describe it as a local/offline development and testing facility.
Documentation will state that GitHub Actions uses real ULB data and fails when
that data is unavailable.

## Non-goals

- Removing the synthetic generator.
- Preventing unit tests from testing synthetic-data behavior.
- Adding a new data mirror, cache service, secret, or external dependency.
- Changing model behavior, dataset schema, pipeline stages, or generated report
  contents beyond their data provenance.
