# -*- coding: utf-8 -*-
"""
A migration manager small enough to read in one sitting.

Plain `.sql` files in `migrations/`, named `NNN_what_it_does.sql`, applied in
filename order, recorded in `schema_migration`. That is the whole design. It is
deliberately not Liquibase (which vog uses, and which is the right answer for a
Grails app with a team): this runs on a 2GB board against a schema with two
tables, and a dependency that needs a JVM to add a column would be absurd here.

Three properties it does have, because their absence is what makes hand-rolled
migration runners hurt:

* **Each migration runs in its own transaction.** A failure half way through
  leaves the database on the last good version, not on a half-applied one.
  Postgres does transactional DDL, so this is real, not aspirational.
* **Applied migrations are checksummed.** Editing a file that has already run
  is the classic way for two machines to silently diverge - dev has the column,
  production does not, and nothing says so. Here it fails loudly on the next
  run and tells you to write a new migration instead.
* **It is idempotent.** Running it when there is nothing to do is a no-op that
  prints so. It is safe to call on every service start, which is what makes
  "did you remember to migrate" stop being a question anybody has to ask.
"""
import hashlib
import os
import sys

MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "migrations")

# Bootstrapped by hand rather than by a migration, for the obvious reason: the
# thing that records which migrations have run cannot itself be one of them.
BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migration (
    id          text        PRIMARY KEY,
    checksum    text        NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now(),
    ms          integer
)
"""


def _checksum(text):
    """Whitespace-insensitive, so reformatting a migration is not a false alarm.

    Reindenting SQL does not change what it does, and a checksum that fires on
    it teaches people to ignore the checksum - which costs you the one case it
    exists to catch.
    """
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def available():
    """Every migration on disk, in apply order, as (id, path, sql)."""
    try:
        names = sorted(f for f in os.listdir(MIGRATIONS_DIR)
                       if f.endswith(".sql"))
    except OSError:
        return []
    out = []
    for n in names:
        p = os.path.join(MIGRATIONS_DIR, n)
        with open(p) as f:
            out.append((n[:-4], p, f.read()))
    return out


def applied(database):
    """{id: checksum} of what has already run, or None if unreachable."""
    if database.execute(BOOTSTRAP) is None and database.last_error:
        return None
    rows = database.query("SELECT id, checksum FROM schema_migration")
    if rows is None:
        return None
    return dict((r["id"], r["checksum"]) for r in rows)


def pending(database):
    """Migrations not yet applied. Raises RuntimeError on a changed file."""
    done = applied(database)
    if done is None:
        return None
    out, drift = [], []
    for mid, path, sql in available():
        c = _checksum(sql)
        if mid not in done:
            out.append((mid, path, sql, c))
        elif done[mid] != c:
            drift.append(mid)
    if drift:
        raise RuntimeError(
            "these migrations have changed since they were applied: %s\n"
            "  Editing an applied migration silently diverges this database "
            "from every other one.\n"
            "  Write a new migration instead, or drop the database if it is "
            "disposable." % ", ".join(drift))
    return out


def migrate(database=None, verbose=True):
    """Apply everything outstanding. Returns the number applied, or -1.

    -1 means the database could not be reached at all, which is different from
    0 (reached, nothing to do) and worth distinguishing at the call site.
    """
    import time
    from . import db as _db
    database = database or _db.db()

    if not database.available():
        if verbose:
            sys.stderr.write("migrate: journal unreachable (%s)\n"
                             % database.last_error)
        return -1
    todo = pending(database)
    if todo is None:
        if verbose:
            sys.stderr.write("migrate: cannot read schema_migration (%s)\n"
                             % database.last_error)
        return -1
    if not todo:
        if verbose:
            print("migrate: up to date (%d applied)"
                  % len(applied(database) or {}))
        return 0

    conn = database.connect()
    n = 0
    for mid, path, sql, checksum in todo:
        t0 = time.time()
        # One transaction per migration. autocommit is on for normal traffic,
        # so it is turned off around the DDL and restored afterwards.
        conn.autocommit = False
        try:
            cur = conn.cursor()
            cur.execute(sql)
            ms = int((time.time() - t0) * 1000)
            cur.execute("INSERT INTO schema_migration (id, checksum, ms) "
                        "VALUES (%s, %s, %s)", (mid, checksum, ms))
            conn.commit()
            cur.close()
            n += 1
            if verbose:
                print("migrate: applied %s (%d ms)" % (mid, ms))
        except Exception as e:
            conn.rollback()
            sys.stderr.write("migrate: FAILED on %s - %s\n"
                             "  nothing from this migration was applied; the "
                             "database is still at the previous version.\n"
                             % (mid, e))
            conn.autocommit = True
            return n
        finally:
            conn.autocommit = True
    return n


def status(database=None):
    """(applied_ids, pending_ids) for `tek memory status`."""
    from . import db as _db
    database = database or _db.db()
    done = applied(database)
    if done is None:
        return None, None
    try:
        todo = [m[0] for m in (pending(database) or [])]
    except RuntimeError as e:
        return sorted(done), ["DRIFT: %s" % e]
    return sorted(done), todo
