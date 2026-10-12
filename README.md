# Private native common-support comparison restart backups

Append-only verified generations. INDEX.json lists checksums, GO identity and
separate source archives. Fetch these LFS objects and use restore_backup.py from the native0139 packet on
the research branch to restore into a fresh run directory. See its README.md for
commands. This preserves committed inference progress, not a running process,
installed runtime or automatic platform restart. No GPU or Actions job is used.
