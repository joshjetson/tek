# -*- coding: utf-8 -*-
"""
The write path: every event TEK considered goes in, including the silent ones.

Recording silences is the part that is easy to skip and expensive to have
skipped. Without them the journal answers "what did TEK say today" but cannot
answer "has it seen anyone", and a run of `spoke = false` rows with a NULL
`decided_ms` is the signature of a brain that is failing rather than one
exercising restraint - the exact ambiguity that cost real debugging time on the
event path.
"""
import json
import sys

from . import db as _db

# psycopg2 2.7 (bionic's apt build) has Json in extras. Imported defensively
# because a checkout without psycopg2 still has to import this module.
try:
    from psycopg2.extras import Json
except ImportError:                                  # pragma: no cover
    Json = None

# UNKNOWN is stored as NULL. Keeping the literal string would make every
# stranger look like one recurring individual to the per-person queries.
UNKNOWN = "UNKNOWN"


def _person(name):
    if not name or name == UNKNOWN:
        return None
    return name.strip().upper() or None


def record(event, said=None, spoke=False, decided_ms=None, model=None,
           database=None):
    """Append one considered event. Returns its id, or None on any failure.

    Never raises: this is called from the voice service's consider-thread, on
    the path that answers a person. A journal write must not be able to turn a
    spoken reply into a stack trace.
    """
    d = database or _db.db()
    extra = {}
    for k in ("faces", "reason", "cooldown", "source"):
        if event.get(k) is not None:
            extra[k] = event[k]
    return d.execute(
        """
        INSERT INTO journal (kind, person, what, heard, said, spoke,
                             model, decided_ms, image_path, extra)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (event.get("kind") or "default",
         _person(event.get("person")),
         event.get("what"),
         event.get("heard"),
         said,
         bool(spoke),
         model,
         decided_ms,
         event.get("image"),
         Json(extra) if Json else json.dumps(extra)),
        returning=True)


def note(person, text, source="manual", ttl_days=None, database=None):
    """Remember a durable fact about someone. Upserts, refreshing the date.

    Re-observing the same fact refreshes it rather than stacking copies, which
    would otherwise crowd everything else out of the retrieval budget.
    """
    d = database or _db.db()
    p = _person(person)
    if not p or not (text or "").strip():
        return None
    expires = None if ttl_days is None else "%d days" % int(ttl_days)
    return d.execute(
        """
        INSERT INTO person_note (person, note, source, expires_at)
        VALUES (%s, %s, %s,
                CASE WHEN %s::text IS NULL THEN NULL
                     ELSE now() + %s::interval END)
        ON CONFLICT (person, note) DO UPDATE
           SET at = now(), source = EXCLUDED.source,
               expires_at = EXCLUDED.expires_at
        RETURNING id
        """,
        (p, text.strip(), source, expires, expires),
        returning=True)


def forget(person, database=None):
    """Drop everything about someone. Pairs with `tek face forget`.

    Deleting a face gallery while leaving a transcript of everything that person
    said would make `tek face forget` a lie. Returns (journal_rows, notes).
    """
    d = database or _db.db()
    p = _person(person)
    if not p:
        return (0, 0)
    j = d.execute("WITH x AS (DELETE FROM journal WHERE person = %s "
                  "RETURNING 1) SELECT count(*) FROM x", (p,), returning=True)
    n = d.execute("WITH x AS (DELETE FROM person_note WHERE person = %s "
                  "RETURNING 1) SELECT count(*) FROM x", (p,), returning=True)
    return (j or 0, n or 0)


def prune(days=400, database=None):
    """Drop journal rows older than `days`, and expired notes.

    Retention is a privacy control, not housekeeping - README section 11 argues
    that short retention is the most effective one available. The default is
    generous enough that "this time last year" still works.
    """
    d = database or _db.db()
    n = d.execute("WITH x AS (DELETE FROM journal WHERE at < now() - %s::interval "
                  "RETURNING 1) SELECT count(*) FROM x",
                  ("%d days" % int(days),), returning=True)
    e = d.execute("WITH x AS (DELETE FROM person_note WHERE expires_at IS NOT NULL "
                  "AND expires_at < now() RETURNING 1) SELECT count(*) FROM x",
                  returning=True)
    return (n or 0, e or 0)


def counts(database=None):
    """Row counts and span, for `tek memory status`."""
    d = database or _db.db()
    rows = d.query(
        """
        SELECT (SELECT count(*) FROM journal)                        AS entries,
               (SELECT count(*) FROM journal WHERE spoke)            AS spoken,
               (SELECT count(*) FROM person_note)                    AS notes,
               (SELECT count(DISTINCT person) FROM journal
                 WHERE person IS NOT NULL)                           AS people,
               (SELECT min(at) FROM journal)                         AS since,
               (SELECT max(at) FROM journal)                         AS latest
        """)
    return rows[0] if rows else None
