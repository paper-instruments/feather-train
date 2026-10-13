# Build sequence

1. Installer and launcher: isolated locked environment, bootstrap, source selection, safe updates.
2. Verification: adversarial caller environment and real model-request/tool execution against a local fake endpoint; add CI coverage.
3. Documentation: update both READMEs with the supported installation and environment contract.
4. Review and simplify, run scoped/full Harness checks, then publish only this change in a child PR based on PR #11.

The user explicitly authorized implementation and a new stacked PR on #11; no additional planning approval is required.

- [x] Installer
- [x] Integration verification and CI
- [x] READMEs
- [x] Review, simplify, verification and publication

Follow-up task list:

- [x] Add task Python selection and shell environment routing.
- [x] Cover CLI/config precedence, invalid interpreters, exact executable selection and installed dual-environment execution.
- [x] Update READMEs and remove private/unmerged installation language.
- [x] Review/simplify, verify and append a scoped commit to PR #12.
