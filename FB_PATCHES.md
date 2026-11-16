# Freebeat patches on Aegra

`freebeatfit/aegra` is a public fork of [aegra/aegra](https://github.com/aegra/aegra)
(Apache-2.0). Each `fb/<upstream version>` branch is the upstream release tag
plus the commits below; releases are tagged `v<upstream version>+fb.<n>` and
publish the `aegra-api` and `aegra-cli` wheels and their `SHA256SUMS` on a
GitHub Release. `fb-super-agent` pins those wheels by release URL and hash, so
installing them needs no credentials. Patches are not sent upstream.

Every release bumps `+fb.<n>`: any change to the patches on the same upstream
version is a new tag. A published release is never rebuilt or re-tagged,
because `fb-super-agent` pins its hashes.

A patch exists only where Aegra contradicts the LangGraph Agent Server, the
Agent Protocol or the LangGraph SDK. Why the fork exists:
`fb-super-agent/docs/adr/0010-run-a-patched-aegra-fork.md`.

| Patch | Fixes | Commit (fb/0.11.0) | Tests |
|---|---|---|---|
| P0 | seq restarts per connection; root lifecycle ids lack `synth:<run_id>` (aegra#751). Cherry-pick of open upstream PR aegra#754. | `fix(api)!: keep thread stream seq and root lifecycle ids stable across connections` | `tests/unit/test_services/test_event_streaming/test_session.py` (`TestSeqAndFilter`, `TestLifecycle`) |
| P1 | Every v2 run attached Updates/Custom/Checkpoints/Tasks; `checkpoints` carried full `get_state()` (C1, C2). Now only a checkpoint-envelope transformer, ordered before its companion `values`. | `feat(api)!: add only the checkpoint envelope transformer to v2 runs` | `test_native_stream.py` (`TestOnlyEnvelopeTransformer`, `TestCheckpointEnvelope`, `TestEnvelopeOrder`) |
| P2 | Replay buffer held every full-state `values`; replay LRANGEd and decoded the whole list (C7). Now one `values` snapshot per namespace in a hash plus a Redis Stream read from the cursor in batches. | `feat(api)!: replay a run from its cursor and keep one values snapshot` | `tests/integration/test_redis_replay_store.py` (needs `AEGRA_TEST_REDIS_URL`), `test_broker_replay.py` |
| P3 | Thread stream replayed every run of the thread, polled the DB every 0.25 s and closed after 30 s idle (C3, C6). Now active runs in full from `since`, the newest settled run compactly, heartbeats until the client leaves, new runs by notice. | same commit as P2 | `test_session.py` (`TestReplayScope`, `TestStaysOpen`) |
| P5 | v2 runs stored the final full state in `runs.output` (C5). Now none; join/wait read the thread's values. | `feat(api)!: store no final state on v2 runs; join/wait read thread values` | `test_run_executor.py` (`TestRunOutputStorage`), `test_run_waiters_output.py` |
| P6 | History ignored `select`, so SDK discovery reads shipped full values. | `feat(api): honor select on thread history` | `tests/integration/test_api/test_threads_history.py` |

P4 (heartbeat, no idle close) is part of P3.

## Known upstream failures

`tests/integration/test_streaming_errors.py` (3 tests) fails on unmodified
`v0.11.0` too; not caused by these patches.

## Rebasing onto a new upstream release

1. `git fetch upstream --tags && git switch -c fb/<new> v<new>`.
2. Cherry-pick the patches above in order. Drop any patch the release already
   fixes, and remove its row here and its contract test in `fb-super-agent`.
3. Set `version = "<new>+fb.1"` in both `libs/*/pyproject.toml`, run
   `make ci-check` (with Postgres and `AEGRA_TEST_REDIS_URL`), push, tag
   `v<new>+fb.1` and let `fb-release.yml` publish the wheels.
4. In `fb-super-agent`, a rebase is a dependency upgrade: it needs the owner's
   approval like any other.
