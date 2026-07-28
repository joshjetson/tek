-- Durable facts about a person, as opposed to the running record in `journal`.
--
-- This is what turns hand-written people.md into something generated. people.md
-- stays supported and stays authoritative - it is the user's own words about who
-- lives here, and a description someone wrote deliberately beats anything
-- inferred. What this adds is the part a person should not have to maintain by
-- hand: that Josh asked about the boiler on Tuesday, that nobody has seen Sam
-- since Friday.
--
-- Deliberately NOT a profile. There is no age, no appearance, no inferred
-- attributes - see README section 11. A note is something that was said or done,
-- attached to a name that a human chose to enrol.

CREATE TABLE person_note (
    id          bigserial   PRIMARY KEY,
    person      text        NOT NULL,
    note        text        NOT NULL,

    -- 'said'    - something they told TEK that is worth keeping
    -- 'topic'   - something they asked about
    -- 'manual'  - written by a human via `tek memory note`
    source      text        NOT NULL DEFAULT 'manual',

    at          timestamptz NOT NULL DEFAULT now(),

    -- Notes decay unless renewed. A household fact from eight months ago is
    -- usually noise, and an assistant that brings one up unprompted is the
    -- uncanny failure this whole feature is meant to avoid. NULL means "keep
    -- indefinitely", which is what `tek memory note` writes: a fact a human
    -- typed in on purpose should not quietly expire.
    expires_at  timestamptz,

    search_fts  tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(person, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(note,   '')), 'B')
    ) STORED
);

CREATE INDEX idx_person_note_search_fts ON person_note USING gin (search_fts);
CREATE INDEX idx_person_note_person ON person_note (person, at DESC);

-- One row per person per note text: re-observing the same fact should refresh
-- it, not stack up twelve copies that then crowd out everything else in the
-- retrieval budget.
CREATE UNIQUE INDEX idx_person_note_unique ON person_note (person, note);
