# Focused recovery prototypes

These seven files preserve the scratch scripts byte for byte. Their original
paths, preserved paths and SHA-256 values are mapped in
`../../recovery-experiments.json`. They support review of the seven case-level
experiments; they are not production entry points.

The scripts expect the original `runs/landing-engine/baseline/` results and
`runs/landing-engine/observation-audit/` paths, and must run from the repository
root. To replay a prototype, restore the seven files to their mapped original
paths first. Prefix and terminal scripts also load the preserved helper scripts
from those paths. They read complete baseline tracks and derive inference
windows from those tracks; reference locations are not inference inputs.

Early prototype runs did not archive a separate imported module snapshot for
every attempt. Their script/output/source/reference hashes are preserved, but
these scripts on a later engine revision need not reproduce the old bytes.
The frozen evaluation harness and final ablation receipts provide the source
and configuration contract for production validation. Prefix, terminal and
dark-component variants did not improve the inspected failure case and remain
rejected experiments.
