# Security

Please report vulnerabilities privately to **support@genuinerobotics.com** (subject:
"gr-brain security") rather than in a public issue. We aim to acknowledge within 3
working days.

Design notes:

* Access tokens (`grb_…`) work on Brain endpoints only, are stored by the platform as a
  hash, and can be revoked on the Brain page. `gr-brain login` saves yours in your user
  config folder (readable only by you on macOS/Linux). Treat it like a password.
* `gr-brain serve` only connects out; it opens no listening port. It receives numbers
  (observations) and returns numbers (actions); it never executes code it is sent.
* Policies you `serve` are your own code running with your permissions — only serve code
  you trust.
