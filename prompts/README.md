# Prompts

Versioned prompt templates. The active version is `prompts.version` in `config/settings.yaml`.

| file | used for |
|---|---|
| `answer_system.<v>.md` | system prompt, identical for C0 and C1 |
| `answer_user_c1.<v>.md` | user message with retrieved passages (C1) |
| `answer_user_c0.<v>.md` | user message without passages (C0, closed-book) |

Every request log records `prompt_version` (e.g. `v1`) and `prompt_sha256` (hash of the three files as
loaded), so an edited file is detectable even if the version id was not bumped. To change a prompt,
copy the three files to the next version (`v2`), edit, and switch `prompts.version`; do not edit a
version that has already been used for reported results.

Placeholders: `{question}` and `{passages}` (C1 only). Each passage is rendered as

```
<passage chunk_id="GWA-1890:29#1" source="Guardians and Wards Act, 1890, Section 29: ...">
...verbatim chunk text...
</passage>
```
