# -*- coding: utf-8 -*-
"""
Connection handling for the memory journal.

The rule this module exists to enforce: **a broken journal must never break the
face.** Memory is an enhancement to what TEK says, not a dependency of its
ability to say anything. Postgres being down, the container being mid-restart,
or the schema being un-migrated all have to degrade to "TEK has no memory right
now" rather than to a stack trace on the path that answers a person.

That is the same shape as `ClaudeBrain.respond` returning None on failure - and
it carries the same hard-won caveat. Returning None *silently* made a broken
brain indistinguishable from a thoughtful silence, and cost real debugging time
(README section 9). So every failure here is swallowed for the caller and
**logged loudly**, with the error kept on `last_error` for `tek memory status`
to show. Quiet degradation is the bug, not the feature.
"""
import os
import sys
import threading
import time

# Imported lazily and tolerantly. A checkout without psycopg2 should still run
# the display, the voice and the ear - it just has no memory.
try:
    import psycopg2
    import psycopg2.extras
    HAVE_PSYCOPG2 = True
except ImportError:                                  # pragma: no cover
    psycopg2 = None
    HAVE_PSYCOPG2 = False

# Host port 5433, not 5432: deploy/docker-compose.yml binds it there so it can
# never collide with another local Postgres, and to loopback only because this
# database records who was home and what was said in the house.
DEFAULT_DSN = "postgresql://tek:tek@127.0.0.1:5433/tek"

# Deliberately short. Every caller is on a path where a person is waiting - an
# event being considered, or a question being answered. A journal that is slow
# to answer is worse than one that admits it is absent, because the cost lands
# as dead air in front of somebody.
CONNECT_TIMEOUT = 3
STATEMENT_TIMEOUT_MS = 2000

# How long to sit out after a failure before trying again. Without this, a
# stopped container turns every single event into another 3-second connect
# timeout, and the latency shows up in front of a person who is waiting.
RETRY_AFTER = 30.0


def dsn():
    """Where the journal lives. Env first, so a service unit can redirect it."""
    return os.environ.get("TEK_DB_DSN", DEFAULT_DSN)


class Database(object):
    """One lazily-opened connection, reopened after failure with a backoff.

    Not a pool. The writers are the voice service's consider-thread and the
    `tek` CLI, neither of which is concurrent with itself, and a pool on a 2GB
    board is memory spent to solve a problem this workload does not have.
    """

    def __init__(self, dsn_=None):
        self.dsn = dsn_ or dsn()
        self.conn = None
        self.last_error = None
        self.failed_at = 0.0
        self._lock = threading.Lock()

    # -- plumbing ----------------------------------------------------------
    def available(self):
        """True if a query could plausibly be run right now."""
        return HAVE_PSYCOPG2 and self.connect() is not None

    def connect(self):
        """A live connection, or None. Never raises."""
        if not HAVE_PSYCOPG2:
            self.last_error = "psycopg2 is not installed"
            return None
        with self._lock:
            if self.conn is not None:
                try:
                    if self.conn.closed == 0:
                        return self.conn
                except Exception:
                    pass
                self.conn = None
            # Back off rather than re-paying the connect timeout per event.
            if self.failed_at and time.time() - self.failed_at < RETRY_AFTER:
                return None
            try:
                c = psycopg2.connect(self.dsn, connect_timeout=CONNECT_TIMEOUT)
                c.autocommit = True
                cur = c.cursor()
                # A runaway query must not hold up an answer. This is belt and
                # braces over the deliberately small LIMITs in recall.py.
                cur.execute("SET statement_timeout = %s", (STATEMENT_TIMEOUT_MS,))
                cur.close()
                self.conn = c
                self.last_error = None
                self.failed_at = 0.0
                return c
            except Exception as e:
                self.last_error = "%s: %s" % (type(e).__name__, e)
                self.failed_at = time.time()
                sys.stderr.write("memory: cannot reach the journal (%s)\n"
                                 % self.last_error)
                sys.stderr.flush()
                return None

    def close(self):
        with self._lock:
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None

    # -- the only two entry points anything else should use ----------------
    def query(self, sql, args=None, default=None):
        """Rows as dicts, or `default` on any failure. Never raises."""
        c = self.connect()
        if c is None:
            return default
        try:
            cur = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, args or ())
            rows = [dict(r) for r in cur.fetchall()]
            cur.close()
            return rows
        except Exception as e:
            self._blame(e, sql)
            return default

    def execute(self, sql, args=None, returning=False):
        """Run a statement. Returns the first column of RETURNING, or None."""
        c = self.connect()
        if c is None:
            return None
        try:
            cur = c.cursor()
            cur.execute(sql, args or ())
            out = cur.fetchone()[0] if returning else None
            cur.close()
            return out
        except Exception as e:
            self._blame(e, sql)
            return None

    def _blame(self, e, sql):
        """Log loudly, and drop the connection so the next call reconnects.

        Dropping matters: psycopg2 leaves a connection in an unusable aborted
        state after an error, and reusing it makes every subsequent query fail
        with a misleading InFailedSqlTransaction that hides the real cause.
        """
        self.last_error = "%s: %s" % (type(e).__name__, str(e).strip())
        sys.stderr.write("memory: query failed (%s)\n  sql: %s\n"
                         % (self.last_error, " ".join(sql.split())[:160]))
        sys.stderr.flush()
        self.close()


# One shared instance. Callers use the module-level helpers in __init__.py.
_DB = None


def db():
    global _DB
    if _DB is None:
        _DB = Database()
    return _DB
