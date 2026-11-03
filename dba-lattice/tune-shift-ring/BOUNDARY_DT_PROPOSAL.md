# Proposed bounded ring-edge refinement

The large-bunch performance probe uses131072 particles and a32³ OPEN mesh.
After about25s it had advanced only14.956fs of the first62.5ps nominal interval.
Resolving every particle's face crossing to coordinate roundoff prevents a
practical many-turn PIC study. The field solver itself completes initial-field
benchmarks in seconds.

## Proposed input change for approval

Add optional RUN attribute `BOUNDARYDT` in seconds, default0:

```text
TRACK, ..., DT=6.25e-11, ...;
RUN, METHOD="PARALLEL", FIELDSOLVER=FS, BOUNDARYDT=6.25e-11, TURNS=64;
```

- `BOUNDARYDT=0` retains today's boundary refinement and all prior precision tests.
- Positive values stop subdivision once an interval is no longer larger than
  BOUNDARYDT (a split can produce intervals between half the bound and the bound).
- It changes only the stopping criterion for existing common Boris/PIC substeps:
  no separate pusher, frozen self fields, particle batching, or CPU transport.
- Geometry's existing maximum-step cap remains enforced independently; reference
  return-plane stopping still localizes the final requested turn.
- Require finite nonnegative values. Reject explicit use outside a supported ring
  boundary-control configuration instead of silently ignoring the setting.
- Log the configured bound and actual trial/rejection counts in run output.

Setting BOUNDARYDT equal to DT yields ordinary common steps (subject to geometry
and final-return caps). Reducing both by two supplies a timestep-convergence
study. The finite boundary approximation trades exact individual face resolution
for measurable discretization error; it does not promise the small-probe
picometre closure at a coarse setting. Bare/charged controls use identical settings.

## Validation before using for tune physics

Unit tests for invalid/zero/positive bounds, cap precedence, duration conservation,
and unchanged zero-default behavior; the existing exact-boundary tests remain.
Compare bare tunes and finite-bunch short-arc results as DT/BOUNDARYDT decrease;
run dedicated MPI consistency separately. User and physics manual entries will
define the parameter and accuracy assumptions. No existing numerical tolerance
is loosened. Source scope: TrackRun attribute parsing/private assignment,
ParallelTracker schedule construction, internal scheduler tests and documentation.

Approval is needed only for introducing this input parameter: AGENTS.md says
"Do not change public APIs without explicit approval." Independent field/RF/
matching work continues while this proposal is reviewed.
