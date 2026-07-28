-- The journal: every event TEK considered, and what it decided to say.
--
-- One wide table rather than a normalised event/utterance/person schema. The
-- read pattern is "give me the most relevant handful of rows for this prompt",
-- always within a budget of a few hundred characters, and joins buy nothing
-- against a table that gains a few dozen rows a day. A household generates
-- roughly 10k rows a year.
--
-- Note what is NOT here: no camera frames, no audio, no transcript of anything
-- that did not pass the wake word. The journal records what TEK was told and
-- what it said back. `image_path` is a path, and the file it points at is
-- overwritten by the next event on purpose.

CREATE TABLE journal (
    id          bigserial   PRIMARY KEY,
    at          timestamptz NOT NULL DEFAULT now(),

    -- arrival | departure | manual | speech | clock | sensor
    -- Matches agent.LEAN's keys, so a row can be replayed as an event.
    kind        text        NOT NULL,

    -- The recognised name, or NULL for UNKNOWN. NULL rather than the string
    -- 'UNKNOWN' so that per-person queries do not accidentally treat every
    -- stranger as one recurring individual.
    person      text,

    what        text,        -- the event description handed to the model
    heard       text,        -- what a person actually said, post-transcription
    said        text,        -- what TEK replied, NULL if it stayed silent

    -- Silence is a first-class outcome everywhere else in this project, so it
    -- has to be recorded as one. Without this column a quiet decision and a
    -- crashed brain look identical in the journal - the exact bug that cost
    -- real debugging time on the event path (README section 9).
    spoke       boolean     NOT NULL DEFAULT false,

    model       text,        -- which brain decided
    decided_ms  integer,     -- how long it took, for the latency HUD strip
    image_path  text,
    extra       jsonb       NOT NULL DEFAULT '{}'::jsonb,

    -- Weighted, STORED, generated. Generated rather than trigger-maintained for
    -- the reason vog's add-fts migration records: a trigger can drift out of
    -- sync with its source columns and nothing says so, whereas a generated
    -- column cannot. Valid here because every term uses the explicit 'simple'
    -- regconfig, which makes to_tsvector immutable.
    --
    -- 'simple' (lowercase + tokenise, no stemming) is also what makes the
    -- 'term:*' prefix queries in recall.py behave: index and query share one
    -- dictionary. Stemming would make 'opened' and 'opening' match, but would
    -- also break prefix matching on partial words from a small recogniser.
    --
    -- Weights: A = what a person said and who they are, because recall is
    -- almost always "what did we say about X". B = TEK's own reply. C = the
    -- event description, which is frequently boilerplate ("someone appeared")
    -- and would otherwise dominate rank on common words.
    search_fts  tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(heard,  '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(person, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(said,   '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(what,   '')), 'C')
    ) STORED
);

CREATE INDEX idx_journal_search_fts ON journal USING gin (search_fts);

-- "the last N things that happened" - the single most common read.
CREATE INDEX idx_journal_at ON journal (at DESC);

-- "the last N exchanges with this person". Partial: rows with no recognised
-- person are never queried this way, and excluding them keeps the index small
-- on a gallery of three or four people against a majority of UNKNOWN rows.
CREATE INDEX idx_journal_person_at ON journal (person, at DESC)
    WHERE person IS NOT NULL;

-- Recap and quiet-hours work in local days, and would otherwise sequential-scan
-- to find "today". Immutable-safe: the expression pins the zone rather than
-- depending on the session's TimeZone setting.
CREATE INDEX idx_journal_day ON journal ((at AT TIME ZONE 'UTC'));
