# Code readiness before GPU experiments

Audited on 28 September 2026. **The project is not code-complete.** GPU access
alone will not make the current checkout run the proposed study. Passing unit
tests cover the implemented components; they do not measure project completion.

| Workflow | Existing executable code | Missing code / decisions |
| --- | --- | --- |
| Local setup and quality | Locked Python package, tests, lint, build, CI workflow | Remote CI evidence |
| Start model services | Validated offline Slurm plan generation | Actual configured jobs, model response checks, training launcher |
| Model interaction | Pure text/function-call response parser | Async transport (dependency deferred), service smoke and tau2 registration |
| Benchmark and tasks | Persona injection on serialized dictionaries | Accepted benchmark pin, task loading, frozen manifests and label/generator implementation |
| Baseline/evaluation | No evaluation runner | Task environment adapter, trajectory recording, metrics, task bootstrap, result export |
| Reward | No reward implementation | Deterministic task adapter, violation/format checks, composer and parity tests |
| SFT | Assistant-only token batching with declared train guard | Teacher rollouts, registry-backed membership/eligibility filtering, dataset export, actual loss/optimizer integration and checkpoint handling |
| Multi-turn RL | Token IDs and response masks can be accumulated | Live token-based generation, tool/user environment steps, verl adapter, algorithm configs and launchers |
| Reproducibility | Config/model/source/seed/hardware records | Driver integration, tracker export, experiment-log generation |
| Presentation | Design documents | Real result figures/tables, report, slides and working demo |

## Work that can proceed without changing scientific definitions

- Implement a model client and strict tool-call response handling; installing a
  new HTTP dependency still requires the separate dependency decision.
- Add model-service smoke checks that explicitly report infrastructure behavior.
- Improve serialization, cancellation, error handling and run lifecycle wiring.
- Preserve the implemented token, provenance and persona invariants while
  connecting these components to their eventual callers.

## Decisions that remain open

ADR-007's HTTPX dependency was also deferred: only the pure parser was authorized
for this implementation slice. Network-client work remains pending.

ADR-006 is still deferred: do not silently adopt its tau2 pin, source-task split
handling, or compliance interpretation. Shared trajectory/reward details and a
trainer dependency also need concrete proposals under AGENTS. Deferring these
does not imply that all other code is already finished.

Once these decisions permit integration, an offline fixture-driven run should
exercise the same actual drivers as a real run, with only network/model/benchmark
boundaries replaced. A separate toy simulation is not evidence that the scientific
pipeline is ready. Real experiments, throughput and dialogue audits remain
separate acceptance steps even after all code exists.
