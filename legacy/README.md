# Legacy single-file agent

`myai-agent.py` here is the single-file Ollama+vLLM agent currently deployed
on hal9000, tanner, mac-mini, lappy, lara-cpu under `/opt/myai-agent/`. It
predates the pip-package layout in `src/` and is kept for operator parity
while the live nodes migrate.

This is the SAME v3-C ECDSA P-256 attestation behaviour as the pip
package -- the two implementations sign envelopes in the same format so the
coordinator's `api/security/attestation.py` validator accepts both.

Spec (v1.6.0):
- ECDSA P-256, key persisted at `/opt/myai-agent/.attest-key.pem` mode 600.
- Pubkey shipped as SEC1 uncompressed (0x04 || X || Y), base64url no-pad.
- Sig shipped as raw r||s (64 bytes), base64url no-pad over
  `f"{agent_id}|{ts}|{nonce}"`.
- Sent on every register + heartbeat + job-complete.
- 409 on register => exit 1 (operator must `--rotate-attest-key`).
- Set `MYAI_DISABLE_ATTESTATION=1` to opt out (legacy 0.5x earnings).

CLI:
```bash
python3 myai-agent.py --show-pubkey
python3 myai-agent.py --rotate-attest-key
```
