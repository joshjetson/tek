# -*- coding: utf-8 -*-
"""
TEK's memory: a Postgres journal of what happened and what was said.

    from tekdromo import memory
    memory.record(event, said="...", spoke=True, decided_ms=740)
    memory.context(event)          -> the block to paste into a prompt

Why this exists, in one line: TEK could recognise you by name and had no idea
you had spoken that morning, which reads as a security camera with a voice
rather than as a presence.

Every call in here is safe to make when Postgres is absent, stopped, or
un-migrated. It degrades to "no memory", loudly logged, never to an exception -
see db.py. Memory improves what TEK says; it is not a precondition of TEK
saying anything.
"""
# Submodules FIRST, under names that cannot be shadowed.
#
# Two of the functions exported below - `migrate` and (previously) `db` - have
# the same name as the module they live in. Exporting them binds the package
# attribute to the FUNCTION, so `from tekdromo.memory import migrate` hands back
# a function where a caller reasonably expected the module, and it fails at
# first use with a bare "'function' object has no attribute '_checksum'" that
# points nowhere near the cause. Worse, had these module imports come second,
# every `from . import db as _db` inside recall/store/migrate would have
# resolved to the function too.
#
# So: import the modules first, alias them explicitly, and let the convenience
# functions shadow only what is safe to shadow. `from tekdromo.memory.migrate
# import x` also still works - that form resolves through sys.modules and
# ignores package attributes entirely.
from . import db as db_mod                                          # noqa: F401
from . import migrate as migrate_mod                                # noqa: F401
from . import recall as recall_mod                                  # noqa: F401
from . import store as store_mod                                    # noqa: F401

from .db import Database                                            # noqa: F401
from .migrate import migrate, status as migration_status            # noqa: F401
from .recall import context, day, recent, relevant, person_context  # noqa: F401
from .store import record, note, forget, prune, counts              # noqa: F401

# Every function in this package takes `database=` as its optional connection
# argument. One name, so a caller never has to remember which module it is in.
database = db_mod.db


def ready(database=None):
    """(ok, detail) - is the journal usable right now?

    Used by `tek memory status`, and by the voice service at startup so the
    answer is logged once rather than rediscovered on every event.
    """
    d = database or _db.db()
    if not d.available():
        return False, (d.last_error or "unreachable")
    done, todo = migration_status(d)
    if done is None:
        return False, (d.last_error or "cannot read schema_migration")
    if todo:
        return False, "%d migration(s) pending: %s" % (len(todo), ", ".join(todo))
    return True, "%d migration(s) applied" % len(done)
