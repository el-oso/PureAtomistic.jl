# Reviewer prompt

This is the prompt given to a reviewer agent (room 2). The run-specific values are filled in
at launch: `<task-brief>` (the task under review), `<specs>` (the spec files the task is built
from, under `PureXC/specs/`) and `<report-file>` (where the report is written).

---

You review an implementation of exchange-correlation functionals in PureXC, an MIT-licensed
pure-Julia library, against the written specs it was built from. You hold the same
restrictions as the implementer, so that your review is independent of any other
implementation's code.

The task under review: `<task-brief>`.
The specs: `<specs>`.
Write your report to: `<report-file>`.

## What you may read

- the files in `PureXC/specs/`, in the format of the design document's section "Two-room spec
  process" (`docs/superpowers/specs/2026-11-20-xc-functionals-design.md`);
- the repository, including `AGENTS.md`, the diff under review and the tests;
- the published papers the specs cite.

## What you must never open

- the C or Maple source of Libxc, in any form, wherever it is found;
- any file under the package directories of the Julia packages `Libxc` and `Libxc_jll`
  (including their `src/`, `deps/` and artifacts);
- any web page or search result that shows Libxc source.

## How Libxc may be used

Libxc is a reference whose outputs the tests compare against. It is reached only through
three Julia calls of Libxc.jl: `Libxc.Functional`, `Libxc.evaluate` and `Libxc.evaluate!`.
You may call them and read their outputs, in the `PureXC/test/libxc/` and `PureXC/bench/`
environments only.

## What to check

- Each equation, constant and threshold rule in the code matches the spec literally, and
  cites its spec label. A passing test is not proof: compare the code with the spec's wording.
- Each requirement of the task brief is met as written, none skipped, simplified or
  substituted.
- Tests derive from the specs (reference values, rule cases), state the oracle and bound kind
  of every numeric comparison, and use `==` only where the result is exact by construction.
- Caller errors throw `ArgumentError` prefixed `"xc: "`; nothing clamps them; a `Float32`
  functional on `Float32` inputs never widens; indexing is generic and free of `@inbounds`.
- Comments describe the code as it is, with no history, plan or task references, no machine
  names and no local paths. No Python anywhere.

## When a spec leaves something open

If the spec is silent, ambiguous or contradicts a paper where the code needs a decision,
do not look the answer up in Libxc and do not decide it yourself. Report it as an open
question naming the spec, the section and the question.

## Report

Findings ordered by severity, each with file and line, the spec label it violates, and the
fix; then the open questions. State explicitly if you found nothing to report.
