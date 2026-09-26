# Community Writer

The Community Writer is an open-source AI Note Writer that proposes
Community Notes through the public [AI Note Writer
API](https://communitynotes.x.com/guide/en/api/overview). It exists to
increase the supply of timely, helpful notes while keeping people on X in
charge of which notes show broadly on X. This repository holds its source
code: a service that continuously fetches X posts eligible for Community
Notes, drafts candidate notes, evaluates the drafts against a series of
automated checks, and submits proposed notes through the API. Proposed
notes are rated by Community Notes contributors like any other note; only
notes that contributors from different perspectives rate as helpful are
shown on posts.

## Designed to reflect the will of the people

Community Notes show on a post when people on X who have historically
disagreed rate them helpful, deciding together through ratings which notes
are helpful. The Community Writer extends that principle to AI note
writing: its judgment about which posts deserve a note, what a good note
looks like, and when a proposed note is unhelpful and should be withdrawn
comes from note requests, ratings and past note outcomes. That shows up in
three ways:

- **People on X determine feed content.** The Community Writer writes only
  on posts in the AI Note Writer API feeds, which are built from the
  [Request a Community
  Note](https://communitynotes.x.com/guide/en/under-the-hood/note-requests)
  feature and other demand and engagement signals from people on X.
- **Contributors drive what it learns.** The Community Writer uses models
  trained on Community Notes data: in particular, which posts ended up
  with a Helpful note and which notes were rated Helpful or Not Helpful.
- **Contributors decide what shows.** Every AI note is scored by the same
  [open-source ranking
  algorithm](https://communitynotes.x.com/guide/en/under-the-hood/ranking-notes)
  as any other proposed note to determine whether it shows. The writer
  also watches ratings on its own notes, and may revise notes that have
  not earned Helpful status or withdraw notes contributors rate poorly.

The Community Writer aims to make good use of contributors' time and
energy. Each component is designed to prioritize the strongest drafts,
optimizing proposed notes to gather ratings that are most likely to be
found helpful by people from [different
perspectives](https://communitynotes.x.com/guide/en/contributing/diversity-of-perspectives),
and therefore to show broadly on X.

## How the pipeline is structured

The service (`src/main.py`) is a queue-based system of four cooperating
asynchronous workers:

- A **producer** fetches eligible posts for each configured API feed on a
  fixed interval, scores them with the Notable Post Model, and
  enqueues them per feed.
- A **consumer** drains the queues and processes each post in an isolated
  subprocess with time and memory bounds. For each post it drafts notes with
  one or more configured writer models, runs the rejector stages, chooses
  misleading-reason tags, and submits notes that pass every stage. Results
  are periodically flushed to parquet files.
- A **config watcher** polls a config directory and applies new TOML
  configurations without a restart.
- A **note status watcher** persists submission history while maintaining
  up-to-date status and ratings for previously submitted notes. The note
  status watcher also applies the Deletion Model to remove notes that are
  underperforming, conserving contributor ratings.

`writer.toml` configures production note writing, including API feeds,
writers, rejector thresholds, submission accounts, daily limits, and
deletion policies.

Packages under `src/`:

| Package | Role |
|---|---|
| `workers/` | The four workers above. |
| `note_writer/` | Drafting: prompts, the LLM client, note-length rules, markdown stripping, suggestion handling, submission orchestration. |
| `rejectors/` | The automated checks a draft must pass: an LLM rejector, a recent-context rejector, a revision rejector, and a screenshot rejector that verifies cited sources against page captures. |
| `cnapi/` | Client for the Community Notes API: eligible posts, note submission, status retrieval, deletion. |
| `data_models/` | Pydantic models for the writer configuration (`writer.toml`) and for runtime data. |
| `notable_post_model/` | Training and inference code for the Notable Post Model used to prioritize posts. |
| `deletion_model/` | Training and inference code for the model behind rating-based deletion policies. |
| `fetcher/`, `browser/` | Screenshot capture of cited source pages used by the screenshot rejector. |
| `utils/` | Startup helpers: environment validation and initialization. |

## Scope

This public code release is designed to include the core Community Writer
implementation, so that it can be audited, critiqued, improved or
replicated. The release includes training code for the Notable Post Model
and Deletion Model, although other components beyond the core Community
Writer implementation are out of scope for this release, such as model
weights, operational infrastructure (e.g. storage management), deploy
scripts, monitoring, testing, etc. Note that production uses a modified
`browser/` package with additional logic supporting screenshot capture.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
