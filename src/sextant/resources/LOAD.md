# What this package loads

This checkout is partial infrastructure. It is not a finished decision compiler.

Loadable from the installed package:

- typed graph checks
- single-writer ledgers and state files
- request/answer handshake with the paid API fallback disabled
- offline calibration tools
- the role-card texts in `role-cards/`
- offline doctor integrity checks

Still partial:

- role cards are question texts, not a calibrated product
- live Jev and generative-model verification (a missing live probe is NOT_RUN)
- crash-safe multi-file transactions (not built yet; a single-writer lock is the current guard)
- product acceptance: gold run, render, verifier, registry
- the decision compiler, policy, execution, and feedback loop

Offline tests and `doctor` PASS lines are not live Jev or generative-model verification.
Release eligibility is the doctor checklist, not the absence of FAIL.
