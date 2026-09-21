# Official hmfast consistency

This folder compares overlapping physics in the licongxu fork against
https://github.com/hmfast/hmfast without requiring the two trees to share an API.

## Live comparison

```bash
PYTHONPATH=src python tests/official_consistency/eval_fork.py /tmp/fork_hmfast.npz
PYTHONPATH=/path/to/official/hmfast/src python tests/official_consistency/eval_official.py /tmp/official_hmfast.npz
python tests/official_consistency/compare.py /tmp/fork_hmfast.npz /tmp/official_hmfast.npz
```

`reference_official.npz` is a frozen dump from official `hmfast` commit
`243f744` (`lcdm:v1`, matched cosmological parameters) used by
`tests/test_official_consistency.py`.
