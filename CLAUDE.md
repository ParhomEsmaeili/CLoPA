# Response formatting

When breaking down findings, review comments, or explanations that involve
multiple distinct points (e.g. code review feedback, bug lists, plan
reviews), use numbered/bulleted steps rather than block text paragraphs.
Each step should be short and scannable, not a dense prose explanation.

## Hand-off

Bug reports get classified and handed off to Daneshvar with tags — the destination
is decided there, not here.

- `bug report` → tags `{ type: bug, source: clopa }`

Phrases live in `.agent-routing.local.yml` (gitignored); see `.agent-routing.example.yml`
for the schema. Attach the tags and hand off.
