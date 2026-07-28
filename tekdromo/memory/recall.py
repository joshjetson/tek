# -*- coding: utf-8 -*-
"""
Retrieval: which handful of past rows belong in this prompt.

Three signals, blended, because any one of them alone is visibly wrong:

* **Recency** alone gives you a scrollback. Ask "what did we decide about the
  boiler" and it hands over the last four things said, none of which are the
  boiler.
* **Relevance** alone gives you a search engine. Say "morning" and it surfaces
  a greeting from March, which is worse than no memory at all because it is
  confidently irrelevant.
* **Person** alone conflates the household.

So: recent turns are taken by time, topical rows are taken by `ts_rank` decayed
by age, the current speaker's rows get a boost, and the two sets are deduped.
The whole thing is capped by a character budget, because everything here is
paid for twice - once in prompt tokens and once in the latency a person is
standing there waiting through.

No embeddings, on purpose. Vector search would need a model resident in the
2GB this board also renders a face out of, to answer questions over a table
that gains a few dozen rows a day. Postgres FTS with a GIN index answers them
in single-digit milliseconds and costs nothing when idle.
"""
import re

from . import db as _db

# 'simple' does not remove stopwords - that is the price of the immutability
# that lets `search_fts` be a generated column. So they come out here instead.
# Without this, "what is the time" matches essentially every row in the table
# and rank ordering becomes noise.
STOPWORDS = set("""
a about after again all also am an and any are as at be because been before
being between both but by can cant come could did do does doing dont down
during each few for from further had has have having he her here hers him his
how i if in into is it its just like me more most my no nor not now of off on
once only or other our out over own same she should so some such than that the
their them then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your
yeah yes ok okay hey tek get got know think thing things really much many lot
""".split())

# Contraction stems. A spoken transcript is full of these, and because the
# parser splits on the apostrophe they arrive as the leading half: "don't" ->
# don, "isn't" -> isn, "what's" -> what. The trailing half (s, t, re, ve, ll)
# is already discarded by the three-character floor. Without these, `don:*`
# and `isn:*` become search terms, which match nothing and burn budget.
STOPWORDS.update("""
don doesn didn isn aren wasn weren won wouldn couldn shouldn haven hasn hadn
can cant thats whats its im ive ill youre theyre lets whos heres theres
""".split())

# A rambling question must not become a forty-term query. The top terms carry
# the topic; the tail is noise that costs index time and dilutes rank.
MAX_TERMS = 8

# Decay constant, in seconds. ts_rank is multiplied by exp(-age / TAU), so a row
# is worth 1/e of its raw rank after this long. Two weeks: long enough that
# "we talked about this last week" works, short enough that a greeting from
# March cannot outrank something from this morning on a common word.
TAU = 14 * 24 * 3600.0

# Rows older than this are not considered at all. This is an optimisation that
# provably cannot change an answer: at 90 days the decay factor is exp(-90/14) =
# 0.0016, so such a row would need 619x the ts_rank of a fresh one to place. It
# was verified rather than assumed - top-4 and top-10 results are identical with
# and without the window across broad, narrow and rare queries.
#
# What it buys is the query plan. Measured on 10,005 rows (a year of household
# traffic) on this board:
#
#   query                        plan                  time
#   broad OR, no window          Seq Scan              76.3 ms
#   broad OR, 90-day window      Bitmap Heap Scan      19.8 ms
#
# Without the window the planner sequential-scans even for a rare term, because
# the whole table is 7MB and fits in shared_buffers - a defensible choice that
# nonetheless makes cost grow with total history rather than with recent
# history. The window bounds the work no matter how many years accumulate.
HORIZON_DAYS = 90

# How much better a row is for being the same person's. Multiplicative and
# modest on purpose: it should break ties between comparably relevant rows, not
# let an irrelevant row from the right person beat a relevant one from anybody.
PERSON_BOOST = 1.6

# The budget everything is rendered into. ~19 characters a second of speech
# means this is not what makes a reply long - it is what makes the prompt big,
# and the prompt is paid for on every single call.
DEFAULT_BUDGET = 1200


def terms(text):
    """Query text -> the words worth searching on.

    The tokenisation deliberately mirrors what Postgres's 'simple' parser does
    to the indexed side, because a query that splits words differently from the
    index silently under-matches. Verified with ts_debug rather than assumed:

        ts_debug('simple', "the boiler's temperature")
          -> the [asciiword], ' [blank], boiler [asciiword], s [asciiword], ...

    So an apostrophe is a SEPARATOR, not part of a word: `boiler's` is indexed
    as `boiler` and `s`. Treating it as one token produced the term
    `boiler's:*`, which to_tsquery turns into the phrase `'boiler':* <-> 's':*`
    - a stricter query than intended, dragging in a junk `s` token. Splitting on
    it and dropping fragments under three characters lands on exactly the tokens
    the index holds.
    """
    if not text:
        return []
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        # Under three characters is below what 'simple' tokenising makes
        # useful, and also disposes of the `s`/`t` halves of contractions.
        if len(w) < 3 or w in STOPWORDS or w in out:
            continue
        out.append(w)
        if len(out) >= MAX_TERMS:
            break
    return out


def tsquery(words):
    """OR of prefix terms, the shape `search_fts` was built to answer.

    OR rather than AND, which is the opposite of vog's ftsSearch and is right
    for the opposite reason. vog is filtering a list a person is looking at, so
    every term must match. This is gathering context for a prompt: an AND over
    a spoken sentence returns nothing almost every time, and a zero-result
    recall is indistinguishable from having no memory.

    Prefix (`term:*`) because the transcript comes from a small recogniser in a
    room and word endings are the first thing it gets wrong.
    """
    return " | ".join("%s:*" % w for w in words)


def relevant(text, person=None, exclude=None, limit=4, database=None):
    """Topical rows, ranked by relevance decayed by age. Never raises."""
    d = database or _db.db()
    words = terms(text)
    if not words:
        return []
    q = tsquery(words)
    ex = list(exclude or []) or [-1]
    rows = d.query(
        """
        SELECT id, at, kind, person, heard, said, what,
               ts_rank(search_fts, to_tsquery('simple', %s)) AS rank,
               extract(epoch FROM (now() - at)) AS age_s,
               ts_rank(search_fts, to_tsquery('simple', %s))
                 * exp(- extract(epoch FROM (now() - at)) / %s)
                 * CASE WHEN %s::text IS NOT NULL AND person = %s
                        THEN %s ELSE 1.0 END AS score
          FROM journal
         WHERE search_fts @@ to_tsquery('simple', %s)
           AND at > now() - (%s || ' days')::interval
           AND NOT (id = ANY(%s))
           AND (said IS NOT NULL OR heard IS NOT NULL)
         ORDER BY score DESC
         LIMIT %s
        """,
        (q, q, TAU, person, person, PERSON_BOOST, q, HORIZON_DAYS, ex, limit),
        default=[])
    return rows or []


def recent(limit=6, max_age_s=600.0, person=None, database=None):
    """The conversation as it stands, newest last.

    This is the durable version of VoiceService.turns. That list is in memory,
    so a service restart mid-conversation used to lose the thread entirely -
    which reads to a person as the device having a stroke rather than as a
    restart.
    """
    d = database or _db.db()
    rows = d.query(
        """
        SELECT id, at, kind, person, heard, said
          FROM journal
         WHERE at > now() - (%s || ' seconds')::interval
           AND (heard IS NOT NULL OR said IS NOT NULL)
           AND (%s::text IS NULL OR person = %s OR person IS NULL)
         ORDER BY at DESC
         LIMIT %s
        """,
        (max_age_s, person, person, limit), default=[])
    return list(reversed(rows or []))


def person_context(person, database=None):
    """When someone was last seen, how often, and what they last brought up."""
    if not person:
        return None
    d = database or _db.db()
    rows = d.query(
        """
        SELECT count(*)                                   AS times,
               max(at)                                    AS last_at,
               extract(epoch FROM (now() - max(at)))      AS ago_s
          FROM journal WHERE person = %s
        """, (person,))
    if not rows or not rows[0].get("times"):
        return None
    out = dict(rows[0])
    notes = d.query(
        """
        SELECT note FROM person_note
         WHERE person = %s AND (expires_at IS NULL OR expires_at > now())
         ORDER BY at DESC LIMIT 3
        """, (person,), default=[])
    out["notes"] = [n["note"] for n in (notes or [])]
    return out


# -- rendering -------------------------------------------------------------

def _ago(seconds):
    """Human, spoken-scale, and deliberately vague past a day.

    "3 days ago" is what a person would say; "on 2026-07-25T14:03Z" is what a
    log would say, and the model reads this aloud.
    """
    if seconds is None:
        return "just now"
    s = float(seconds)
    if s < 90:
        return "a moment ago"
    if s < 3600:
        return "%d minutes ago" % (s / 60)
    if s < 7200:
        return "an hour ago"
    if s < 86400:
        return "%d hours ago" % (s / 3600)
    if s < 172800:
        return "yesterday"
    if s < 8 * 86400:
        return "%d days ago" % (s / 86400)
    if s < 60 * 86400:
        return "%d weeks ago" % (s / (7 * 86400))
    return "months ago"


def _line(row, with_age=True):
    who = row.get("person") or "someone"
    when = _ago(row.get("age_s")) if with_age else None
    bits = []
    if row.get("heard"):
        bits.append('%s said "%s"' % (who, row["heard"].strip()))
    if row.get("said"):
        bits.append('you replied "%s"' % row["said"].strip())
    if not bits and row.get("what"):
        bits.append(row["what"].strip())
    if not bits:
        return None
    body = "; ".join(bits)
    return "  - %s: %s" % (when, body) if when else "  - %s" % body


def context(event, budget=DEFAULT_BUDGET, database=None):
    """The memory block for a prompt, or "" if there is nothing worth saying.

    Returning "" rather than a header with nothing under it matters: an empty
    "WHAT YOU REMEMBER" section reads to the model as "you remember nothing
    about this person", which is a stronger and usually wrong claim than
    staying quiet about memory altogether.
    """
    d = database or _db.db()
    if not d.available():
        return ""

    person = event.get("person")
    if person == "UNKNOWN":
        person = None
    probe = event.get("heard") or event.get("what") or ""

    now_rows = recent(limit=6, person=person, database=d)
    seen = set(r["id"] for r in now_rows)
    past = relevant(probe, person=person, exclude=seen, limit=4, database=d)
    who = person_context(person, database=d) if person else None

    out, used = [], 0

    if who:
        head = "You have seen %s %d time%s before, last %s." % (
            person, who["times"], "" if who["times"] == 1 else "s",
            _ago(who.get("ago_s")))
        out.append(head)
        used += len(head)
        for n in who.get("notes") or []:
            line = "  - %s" % n
            if used + len(line) > budget:
                break
            out.append(line)
            used += len(line)

    # Topical recall before the running transcript: the model already gets the
    # live conversation from VoiceService.turns, so the rows it does NOT
    # otherwise have are the ones worth spending the budget on first.
    if past:
        out.append("Relevant things from before:")
        for r in past:
            line = _line(r)
            if not line or used + len(line) > budget:
                break
            out.append(line)
            used += len(line)

    if now_rows and used < budget:
        out.append("Earlier in this session:")
        for r in now_rows:
            line = _line(r, with_age=False)
            if not line or used + len(line) > budget:
                break
            out.append(line)
            used += len(line)

    if not out:
        return ""
    return "WHAT YOU REMEMBER:\n" + "\n".join(out)


def day(hours=24, limit=40, database=None):
    """Everything notable in the last `hours`, oldest first - backs `tek recap`."""
    d = database or _db.db()
    rows = d.query(
        """
        SELECT id, at, kind, person, heard, said, what, spoke,
               extract(epoch FROM (now() - at)) AS age_s
          FROM journal
         WHERE at > now() - (%s || ' hours')::interval
         ORDER BY at ASC
         LIMIT %s
        """, (hours, limit), default=[])
    return rows or []
