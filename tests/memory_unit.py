#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The memory journal: retrieval logic without a database, then the real thing.

Split deliberately. Everything that decides WHAT to retrieve - stopwords, query
building, decay, budget, rendering - is pure and runs anywhere, because that is
the part with judgement in it and it should not need a container to test. The
Postgres half is skipped with a loud message when the journal is unreachable,
rather than failing, so a contributor on a laptop still gets a useful run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

# `migrate_mod`, not `migrate`: the latter is the convenience FUNCTION that the
# package exports, not this module. See the note in memory/__init__.py.
from tekdromo import memory                            # noqa: E402
from tekdromo.memory import recall                     # noqa: E402
migrate = memory.migrate_mod

FAIL = []


def check(label, got, want):
    ok = got == want
    if not ok:
        FAIL.append("%s: got %r want %r" % (label, got, want))
    return ok


def assert_true(label, cond, detail=""):
    if not cond:
        FAIL.append("%s%s" % (label, (" - " + detail) if detail else ""))


# -- term extraction -------------------------------------------------------
# The whole reason this exists: 'simple' does not strip stopwords, so without
# it "what is the time" matches most of the table and rank becomes noise.
check("stopwords removed",
      recall.terms("what is the time and where is the boiler"),
      ["time", "boiler"])
check("wake word is a stopword",
      recall.terms("hey tek what about the kettle"), ["kettle"])
check("short words dropped", recall.terms("go to my pc ok"), [])
check("dedupes", recall.terms("boiler boiler boiler heating"),
      ["boiler", "heating"])
assert_true("term count is capped",
            len(recall.terms(" ".join("word%d" % i for i in range(40))))
            <= recall.MAX_TERMS)
check("empty in, empty out", recall.terms(""), [])
check("punctuation does not leak", recall.terms("boiler's temperature!"),
      ["boiler", "temperature"])

# -- query building --------------------------------------------------------
# OR, not AND: an AND over a spoken sentence returns nothing almost every time,
# and an empty recall is indistinguishable from having no memory at all.
check("prefix OR query", recall.tsquery(["boiler", "heating"]),
      "boiler:* | heating:*")
check("empty terms make an empty query", recall.tsquery([]), "")

# -- the spoken-scale clock ------------------------------------------------
# This text is read ALOUD, so it has to be what a person would say.
check("seconds", recall._ago(10), "a moment ago")
check("minutes", recall._ago(600), "10 minutes ago")
check("hours", recall._ago(4 * 3600), "4 hours ago")
check("yesterday", recall._ago(30 * 3600), "yesterday")
check("days", recall._ago(3 * 86400), "3 days ago")
check("weeks", recall._ago(21 * 86400), "3 weeks ago")
check("vague past two months", recall._ago(200 * 86400), "months ago")
check("no timestamp", recall._ago(None), "just now")

# -- decay ------------------------------------------------------------------
# The horizon is only sound if rows at it are already worthless.
import math                                            # noqa: E402
at_horizon = math.exp(-(recall.HORIZON_DAYS * 86400.0) / recall.TAU)
assert_true("horizon rows cannot outrank fresh ones", at_horizon < 0.01,
            "decay at horizon is %.4f" % at_horizon)
assert_true("decay is meaningful within the window",
            math.exp(-(7 * 86400.0) / recall.TAU) > 0.5)

# -- rendering --------------------------------------------------------------
row = {"person": "JOSH", "age_s": 7200, "heard": "is the boiler on",
       "said": "Yes, since six.", "what": None}
line = recall._line(row)
assert_true("line names the person", "JOSH" in line)
assert_true("line carries both halves",
            "boiler" in line and "since six" in line.lower())
assert_true("line is dated", "hours ago" in line)
assert_true("undated form omits the age",
            "ago" not in recall._line(row, with_age=False))
check("a row with nothing in it renders nothing",
      recall._line({"person": None, "age_s": 1}), None)

# -- migration checksums ----------------------------------------------------
# Whitespace-insensitive: reindenting SQL must not fire the drift alarm, or
# people learn to ignore the alarm.
a = migrate._checksum("SELECT  1")
b = migrate._checksum("SELECT 1")
c = migrate._checksum("SELECT 2")
check("reformatting does not change the checksum", a, b)
assert_true("a real edit does change it", a != c)
mig = migrate.available()
assert_true("migrations are discoverable", len(mig) >= 2)
assert_true("migrations sort into apply order",
            [m[0] for m in mig] == sorted(m[0] for m in mig))

# -- graceful degradation ---------------------------------------------------
# The invariant the whole module exists to hold: no journal, no exception.
from tekdromo.memory.db import Database                # noqa: E402
dead = Database("postgresql://nobody:nobody@127.0.0.1:1/nope")
check("a dead database is not available", dead.available(), False)
check("query returns the default", dead.query("SELECT 1", default="DEFAULT"),
      "DEFAULT")
check("execute returns None", dead.execute("SELECT 1"), None)
assert_true("the failure is recorded for `tek memory status`",
            bool(dead.last_error))
check("context() on a dead journal is empty, not an error",
      recall.context({"kind": "speech", "heard": "anything"}, database=dead), "")
check("relevant() on a dead journal is empty",
      recall.relevant("boiler", database=dead), [])

# -- the real database, if it is there --------------------------------------

live = memory.database()
if not live.available():
    print("memory_unit: SKIPPED the Postgres half (%s)" % live.last_error)
    print("             start it with: cd deploy && docker-compose up -d")
else:
    ok, detail = memory.ready(live)
    assert_true("journal is migrated", ok, detail)

    before = (memory.counts(live) or {}).get("entries", 0)
    ev = {"kind": "speech", "person": "TESTPERSON",
          "heard": "what did we say about the flux capacitor", "faces": 1}
    rid = memory.record(ev, said="It needs 1.21 gigawatts.", spoke=True,
                        decided_ms=700, model="test", database=live)
    assert_true("record() returns an id", bool(rid))
    after = (memory.counts(live) or {}).get("entries", 0)
    check("the row landed", after, before + 1)

    # A silence must be recorded too, or the journal cannot tell restraint
    # from a brain that is failing.
    memory.record({"kind": "arrival", "person": "TESTPERSON"},
                  said=None, spoke=False, decided_ms=900, database=live)

    hits = memory.relevant("flux capacitor", person="TESTPERSON",
                           database=live)
    assert_true("FTS finds it", any("flux" in (h["heard"] or "") for h in hits))
    assert_true("it is scored", all(h.get("score") is not None for h in hits))

    blk = memory.context({"kind": "speech", "person": "TESTPERSON",
                          "heard": "remind me about the flux capacitor"},
                         database=live)
    assert_true("context mentions the person", "TESTPERSON" in blk)
    assert_true("context stays inside its budget",
                len(blk) < recall.DEFAULT_BUDGET + 200,
                "block was %d chars" % len(blk))

    # A note, and the upsert that stops the same fact stacking up.
    memory.note("TESTPERSON", "likes the heating off", database=live)
    memory.note("TESTPERSON", "likes the heating off", database=live)
    pc = memory.person_context("TESTPERSON", database=live)
    check("the note is not duplicated", len(pc["notes"]), 1)

    # forget() must take the transcript too, or `tek face forget` is a lie.
    j, n = memory.forget("TESTPERSON", database=live)
    assert_true("forget removed the journal rows", j >= 2, "removed %d" % j)
    check("forget removed the notes", n, 1)
    check("nothing is left", memory.person_context("TESTPERSON", database=live),
          None)

if FAIL:
    print("MEMORY FAIL")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print("MEMORY OK")
