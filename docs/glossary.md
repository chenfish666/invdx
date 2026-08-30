> **English** · [繁體中文](glossary.zh-TW.md)

[← back to docs index](README.md)

# Glossary: what the Chinese side calls each term

## This page holds decisions, not observations

This page holds **rulings**: how a concept must be written on the Chinese
side, which English term it binds to, and why. It does **not** hold claims
about what any file currently says — those are recomputed on every run of
`make bilingual` (`scripts/check_bilingual.py`) instead of being written down.

The split was forced by three revision rounds. This page causes the other six
Chinese documents to be revised, and a revision is exactly what invalidates a
recorded observation, so **the page rots at its own revision rate**:

- Round one: a `gate` ruling cited a sentence in `env.zh-TW.md` as evidence —
  a sentence rewritten in the very revision this page prompted.
- Round two: the anchoring rule below was introduced, requiring filename plus
  section name. A section name written down in that same round had never
  existed in the file it pointed at — **the form the new rule called safest
  was already broken in the round that introduced it**, and nothing was
  positioned to notice.
- Round three: a Chinese rendering this page bans survived nowhere in the tree
  except in the row banning it, while the page still claimed a named file was
  using it.

Every round fixed the rot found at the time and grew new rot. Fixing the
instances one more time only buys the next round, so what changed is the
division of labour:

| Written here (rulings; they do not rot) | Recomputed by `make bilingual` (observations; they do) |
|---|---|
| concept, prescribed Chinese form, bound English term (the fourth column, "why", is a comment and is not maintained — see below) | whether every filename and section name cited here resolves |
| which renderings are banned | whether a banned rendering still occurs anywhere in either language tree |
| which Chinese form binds which English join key | whether the English term is present wherever the Chinese form is used |

The test is easy to apply. A reason may say "the obvious Chinese rendering of
*process* also means a travel itinerary" — that is **a fact about Chinese**,
and no edit to any file can falsify it. A reason may not say
"`optimize.zh-TW.md` currently writes X" — that is **a description of another
file's present state**, and that file is being revised because of this page.

The checker only does what a machine can do: string matching and anchor
resolution. It cannot read meaning, and it cannot tell whether a symbol was
really defined where it first appears. That still needs a cold reader.

### The first three columns are the content; the fourth is a comment

A ruling needs a reason, and a good reason naturally reaches for the corpus.
Two earlier rounds each stated in writing that the reason column may not
describe the corpus's present state, and both rounds wrote one anyway. That is
not sloppy execution, it is a fight with how writing works — so this round
stops issuing the same prohibition a third time and grades the columns instead:

- **The first three columns (concept / English original / prescribed Chinese
  form) are this page's operative content.** The ban scan, the join-key
  binding, the per-row column counts and the ruling-count floor all act on
  them, and `make bilingual` recomputes the lot on every run.
- **The fourth column, "why", is an explanation written for people, and it is
  not maintained.** It may be out of date, and **it is not to be read as a
  current claim of this project**. Finding it stale is **not** something you
  have to fix — you may, but nobody is obliged to, and leaving it is not a
  defect.

The reason is the record at the top of this section: every round fixed the rot
the reason column had at the time, and every round grew new rot. It rots at the rate
the corpus is revised, and no mechanical check can see the fourth column, so
that rot only accumulates. Pretending it is maintained is worse than declaring
honestly that it is not — **a stale explanation that looks maintained misleads
more than a stale explanation labelled unmaintained**, because the next reader
takes the first one for a current claim.

**Two exceptions stay in the operative content**, because they are not
explanation, they are claims:

- **Numbers and formulas** in the reason column are technical claims of this
  project. Sitting in the fourth column does not exempt them from correction:
  if one contradicts a ruling in `RETRACTIONS.md`, that is a defect and it
  gets fixed.
- **Corpus citations written as anchors** are still verified by the anchor
  check. That check is not relaxed by this demotion: an anchor is something a
  machine can see, and what a machine can see stays in the operative content.

## How this list was built

This is not a list of terms someone thought a glossary ought to contain. It
was **grown from evidence**.

After the six Chinese counterparts ([`env`](env.md),
[`dependencies`](dependencies.md), [`journal`](journal.md),
[`optimize`](optimize.md), [`RETRACTIONS`](RETRACTIONS.md),
[`tolerance`](tolerance.md)) were written, each was handed to a reader who had
not written it and had not read the English original, and every one of them was
asked the same question: **"list every term whose meaning you had to guess."**
Ninety-five came back. This page is what survived from those ninety-five.

That procedure is the point. An author cannot see the terms they coined —
those words grew inside the author's own context, where each one is obvious.
**This is a property of the position, not of effort.** So the entry queue has
to be a fresh reader's guess-list; it cannot be the author's recollection.

One thing an English-side maintainer should take from this page:
**renaming an English term moves a bound Chinese term with it.** The second
column is the join key, and `make bilingual` now enforces it — rename
`production-scale` in English and the Chinese ruling stops matching, loudly,
in the section where it stopped matching.

One maintenance caveat: this page is itself the same content written twice,
once per language. `make bilingual` can check that the two halves line up
structurally. It cannot check that they mean the same thing.

### Anchoring rule: how this page points at other files

This page causes other documents to be revised, and what a revision changes is
sentences. **So an anchor cannot be a sentence.**

There is one spelling, because a machine has to parse it:

- the filename goes in backticks, e.g. `env.zh-TW.md` — resolved relative to
  the citing page's directory, so the top-level README is `../README.md`,
  otherwise it resolves to `README.md` in this folder;
- the section name goes in Chinese angle brackets, immediately after the
  filename;
- for a section of the same page, the filename is omitted — as the Chinese
  half does when it points at its own last section,
  `glossary.zh-TW.md`〈怎麼維護〉.

`make bilingual` resolves every one of them: the file must exist and the
section name must really be a heading in it. A failure exits 1 and prints the
closest headings in that file, so the next person has something to work with.

**Never a verbatim quotation as evidence.** A quotation cannot be resolved,
and it is guaranteed to be rewritten by the next revision.

The one exception is **the ruled word itself**. Those are not anchors, they
are the subject of the ruling; they appear inside a "do not use 「…」" clause,
which is exactly how the checker recognises the ban list.

**The "why" column points at the corpus in that same spelling, or not at
all.** Whenever a reason mentions this repo's corpus — what some file says,
where two words appear together, whose section heading a phrase belongs to —
it has to be written as an anchor in the spelling above (the backticked path,
then the section name in Chinese angle brackets), so the anchor check can see
it. An observation that will not fit that form ("the two appear near each
other in some document", "the whole document is about X") is not badly
phrased, it is the wrong thing to write: it is a present state, this page's
own revisions falsify it, and nothing is positioned to notice. Delete it; keep
the half of the reason that stands on its own.

This does **not** apply to facts about the Chinese language. "The obvious
Chinese rendering of *process* also means a travel itinerary" is not a corpus
observation — no edit to any file can falsify it — so it needs no anchor. The
test is one question: **does verifying this sentence require reading this
repo, or only knowing Chinese?**

## 1. One concept, two Chinese names

This section is worth more than the others. A hard word costs a reader one
stumble; **the same concept under two different names in two different files
makes the reader believe there are two concepts** — and that error grows with
every file added.

A third column beginning with 「 declares a **join key**: that Chinese form is
bound to the English term in column two, and the checker verifies it section
by section. A third column beginning with a backtick means the ruling is to
keep the English word, so there is no Chinese join key.

| Concept | English original | Prescribed Chinese form | Why (mechanism and collision) |
|---|---|---|---|
| painting a design onto the design pixel grid | rasterize | `rasterize` | These docs are about gratings throughout, and the obvious Chinese rendering of *rasterize* is built on the word for *grating*. Two renderings are banned: the literal one, and a second coined phrase meaning "re-discretise into geometry" |
| the deployed, actually-used thing | production | 「正式運轉」, glossed with *production* at first use | This row asks **whether something is the set-up actually in service** (that driver, that dependency group), not how big it is. The banned rendering means a factory production line — actively misleading in a project about SOI/DUV fabrication |
| how big the run is | production-scale | 「正式規模」, glossed with *production-scale* at first use | This row asks about **size**, as opposed to the small smoke and toy inputs; the 0.8 ps, θ=10 pair is not one of those — it is an instance of the size this term names (see `RETRACTIONS.md`〈2026-08-17 — Gradcheck failure mechanism misattributed to float32 cancellation〉). **Not the same concept as the row above, and not interchangeable.** A rendering fusing the two is banned: a fused term is harder to find than two competing ones, because it reads as if it had already been unified |
| the differentiable version of the same CE chain | jnp twin | 「jnp 可微分版本」 | Both Chinese words for *twin* read as *digital twin*, and nothing here is a twin of a physical thing. The metaphor is dropped rather than translated |
| `GRADCHECK_MIN_REL_GRAD`: voxels below a fraction of peak gradient are not sampled by gradcheck | signal floor | 「訊號下限」 | The threshold gets exactly one name. A second name for the same constant makes a reader believe there are two filters |
| acceptance gate, coded G0–G5 | gate | `gate` | Only one Chinese word is allowed as a gloss. The Chinese for *threshold* is reserved for thresholds: that one Chinese word reads equally well as a pass/fail checkpoint and as a numeric limit, so each sense is pinned to a different word |
| two unrelated senses: gradient rematerialisation (`num_checkpoints`) and resume-after-kill (`opt_state.npz`) | checkpoint | `checkpoint` | Both senses are upstream usage, and no Chinese word covers both without misleading. **Untranslated, and every document must say which sense at first use** |
| a spack package definition file | recipe | `recipe` | Kept in English so it matches `spack.yaml` and `spack.lock`, the strings you actually meet in tool output. The banned Chinese rendering means a cooking recipe or a prescription, and it blurs into "default settings" |
| one execution, and the directory it lands in | run | `run` / `run` directory | The Chinese word for *round* reads as loop iteration count, while a run is one execution |
| the minimal "does it start at all" check, matching `--tag smoke` | smoke test | smoke 測試 | Keeping `smoke` is what matches the flag. The fully translated form collides with the literal fire-safety smoke test, which Chinese already has |
| the component deciding how GPU memory is handed out | allocator | 「記憶體配置器」, glossed with *allocator* at first use | The Chinese word first chosen reads as *configurator*, inside a passage entirely about memory limits — the misreading destroys the argument. The word for *memory* is mandatory |
| the XLA component deciding buffer layout | planner | 「記憶體規劃器」, glossed with *planner* at first use | Same reason as the row above, plus: the two must be visibly the same kind of thing. Leaving `planner` bare in English suggests it is unrelated to the allocator |
| a known trap that trips people | pitfall | 坑 | One word throughout. Switching to a bare English `gotcha` mid-document makes readers look for a second concept |

## 2. Collisions: the Chinese word already means something else

Collisions are the dangerous class, because **the reader does not notice they
misread**. The sentence parses; it just says something else. The fourth column
is that something else.

| Concept | English original | Prescribed Chinese form | What it collides with |
|---|---|---|---|
| two senses: the L0 GPU driver, and the program that runs the optimisation loop | driver | `driver`, sense stated at first use | In Chinese this first means *device driver*, which pulls both senses in column one towards the GPU one; context alone is not enough |
| a pytree leaf array | channel | "pytree leaf array", spelled out | In a photonics project, *channel* reads as a waveguide or wavelength channel |
| counting it out from the mechanism, citing no prior conclusion | from first principles | 「從頭逐項清點」 | The Chinese phrase for *first principles* means *ab initio* electronic-structure calculation in exactly this field — a hard pre-existing meaning |
| the whole computational region of the FDTD simulation | scene | 「模擬場域」, glossed with *scene* at first use | The literal Chinese for *scene* means a scene in a story or a social setting |
| the eroded / nominal / dilated density fields | three-field | "the three density fields (eroded/nominal/dilated)" | The Chinese word for *field* collides with the electromagnetic fields the documents are about |
| `version()` / `depends_on()` / `variant()` in a spack recipe | directive | "declaration", glossed with *directive* | The Chinese word for *command* first means the line you type at a shell; reusing it for a recipe's `version()` makes one word carry two senses |
| the computing centre the machine lives in | site | "computing centre / machine room", glossed with *site* | The literal Chinese for *site* means a railway platform or a website, so "site configuration" reads as "website configuration" |
| two senses: a `make` target, and spack's `target=` CPU microarchitecture | target | "Makefile target", distinguished from spack `target=` | Same word, two senses, and Chinese has a single word for both — translate it and the distinction is gone. Each is labelled at first use |
| one independently executing program instance | process | `process` | The Chinese word first means a travel itinerary, so the English is kept; and the argument in `dependencies.md`〈What the licenses add up to〉 rests entirely on whether code enters the same process, so misreading it misreads that whole section |
| the shape of the CE spectral peak, and the same word in the field name `ridge_lam_um` | ridge | 「頻譜峰形」 | The literal Chinese means a mountain ridge; and the `ridge` here is not a ridge waveguide either |
| checking a number against an independent source | reconcile | "check against / verify" | The Chinese word first chosen is accounting vocabulary |
| the source tarball GitHub builds from a tag | archive | `archive` | The Chinese sense of *archive* is closer to *seal away*, and Chinese would also call a dist tarball a "compressed file" — the two have to stay distinguishable |
| the schedule by which β rises over iterations | schedule | `schedule` | The Chinese word for *schedule* first means job scheduling, the Slurm sense; this row is about how β ramps, and translating it points the reader at the other thing |
| Boost Software License | BSL-1.0 | `BSL-1.0`, glossed "(Boost)" at first use | Visually near-identical to **BUSL** (Business Source License), which is **not** open source. Misreading a licence identifier inverts the licensing conclusion |

## 3. Terms deliberately kept in English

Here the prescribed Chinese form is the English word itself — the ruling is
**do not translate**. Usually because Chinese has no settled equivalent, so
inventing one adds a word that itself needs explaining; or because it is the
exact string you will meet in `uv.lock`, in tool output and in upstream docs,
and translating it makes it unsearchable.

| Concept | English original | Prescribed Chinese form | Why it is not translated |
|---|---|---|---|
| copy a specific upstream release into this repo and maintain it here instead of installing it | vendor | vendor (as a verb) | Chinese has no verb "to vendor", and the noun means *supplier* — an inversion, since the point is that **we** took the code in |
| a spack virtual package: an abstract name such as `mpi` satisfied by several real implementations | virtual | virtual | Translating it yields "virtual machine" or "virtualenv" |
| a package already present on the system that spack adopts instead of building | external | external | A spack term; translated, it no longer matches the key in `spack.yaml` |
| the specs you named yourself under `specs:` in `spack.yaml`, as opposed to those pulled in transitively | root spec | root spec | Both words are spack vocabulary; translating either half loses the correspondence |
| solving abstract requirements into fixed versions and build options, and the solver that does it | concretize / concretizer | concretize / concretizer | The result lands in `spack.lock`. Verb and noun have to be introduced together — glossing only one leaves readers assuming the other is something else |
| spack build options (`+python +mpi`) | variant | variant | The obvious Chinese word first means a biological or viral variant |
| an optional dependency group declared in `pyproject.toml`; `--extra gpu` installs that group | extra | extra | A Python packaging term. Translated, it reads as "an extra GPU" |
| the package named in `[build-system].requires` that turns source into a wheel | build backend | build backend | It is itself a download, which is why it trips people on offline hosts; translated, upstream docs become unsearchable |
| links the project into the environment in place instead of copying it | editable install | editable (`-e .`) | The flag is literally `-e`; a translated name no longer matches it |
| a file declaring what a package contains | manifest | manifest | No Chinese word covers both the packaging and the spack sense |
| the conda identifier distinguishing different builds of the same version | build string | build string | It appears verbatim as a field in conda output |
| the file that pins every package to an exact version (`uv.lock`, `spack.lock`) | lockfile | lockfile | The filename is `.lock` |
| an MPI process index; `mpirun -np 2` shows 2 of them | rank | rank | The Chinese word for *rank* is the linear-algebra rank — a hard collision |
| the state a JAX `while_loop` / `scan` carries between steps | carry | carry | It is the API's parameter name |
| JAX handing an input buffer to the output in place to save a copy | donation | donation | The literal Chinese reading is "a charitable donation"; it is the *donate* in `donate_argnums` |
| the layer built from source on this machine rather than installed from wheels (L2: Meep, MPI, HDF5) | native | native (layer) | It is part of the heading `dependencies.md`〈Native layer — the Meep environment〉; translating it breaks the correspondence with the other half's heading |
| the binary-level interface contract | ABI | ABI | The acronym is used untranslated in Chinese technical writing too; a mismatch breaks even when version numbers look compatible |
| a C header file | header | header | Written "系統標頭檔 (header)"; the bare Chinese word collides with HTTP headers and file headers |
| an invdx problem definition — geometry, measurement and acceptance — living in `src/invdx/problems/` | problem | problem | It is the module and directory name |
| the finite-difference check on the adjoint gradient | gradcheck | gradcheck | It is also the function name |
| the morphological operations; here they stand for under-etch and over-etch | dilation / erosion | dilation / erosion | The Chinese names carry an image-processing sense; here they mean process deviation |
| the set of corners | ensemble | ensemble | Best written out as "worst-case or softmin over the three corners" |

## 4. Calques: translated, but the Chinese does not parse

Section 3 is "do not translate". This one is "translated badly". The shared
symptom: **the Chinese characters on their own do not carry the original
meaning**, so the reader reverse-engineers it from context — with no signal
when the reverse-engineering goes wrong.

The bad renderings are recorded as bans, not as "file X used to write it that
way". The latter describes another file's state and rots; the former is a
ruling that stands on its own — and `make bilingual` re-confirms on every run
that none of them has come back.

| Concept | English original | Prescribed Chinese form | Why the calque fails |
|---|---|---|---|
| a lot hangs off it; touching it breaks a chain of other things | load-bearing | "bearing weight / a lot rests on it" | The banned word means *strenuous* or *a demanding role*, not "much is attached to it"; "a private API is load-bearing" comes out ungrammatical |
| one more thing to maintain and keep in step with upstream | surface | "one more thing to maintain" | A single character meaning *face/side* carries none of the sense |
| upstream supports it directly; you do not have to add it | first-class | "officially supported / present out of the box" | The banned rendering means *first grade*, which in Chinese is a rating or a priority level |
| a directory full of wheels, the kind `--find-links` points at | flat index | "a directory full of wheels (the `--find-links` kind)" | The literal rendering is readable only by reverse-engineering from `--find-links`, and a wrong guess gives no signal |
| tucked under something, not at the level you expect | nested | "tucked under / nested under" | The banned rendering uses the Chinese noun for *nest* as a verb, which it is not |
| using a version number as a stand-in for something else | proxy | "using a version number as a stand-in" | The banned rendering reads as *proxy server*, and its second word also means both *metric* and *pointer* |
| the two layers drift further apart over time | drift apart | "the two layers drift further apart" | The banned rendering is an invented verb |
| taken for granted, never actually verified | assumed | "taken for granted" | The banned rendering is read first as *default value* — the opposite meaning |
| the whole curve moves up or down, shape unchanged | shift the level | "the whole curve shifts up or down" | The banned rendering is real engineering Chinese but too abstract here; the reader cannot tell the whole curve is meant |
| this one calls for a different question | Ask a different question | "this one calls for a different question" | The banned rendering means "this cell", which presumes a table the reader is looking at; there is none |
| the parameter controlling erosion / dilation | knob | "the parameter controlling erosion/dilation" | The banned rendering sends a Chinese reader looking for a physical knob |
| the two differences must be exactly 0, no numerical drift permitted | bit-exact contract | "these two differences must be exactly 0, no numerical drift permitted" | "Contract" on its own is opaque; what is meant is a checkable equation |
| the min-width rule the filter was built from | the min-width rule the filter was built from | "the rule the filter was originally built from" | The literal reading is "a rule that has been filtered", which is meaningless — rules are not filtered |
| where the number this page opens with comes from | the motivating number | "where the number this page opens with comes from" | The banned compound is not a Chinese word; it has to be expanded into a clause |
| the README guardrail that config is the single source of truth and scripts never hardcode numbers | the README's "Hard-won guardrails" | `../README.md`〈Hard-won guardrails (encoded in `engines/conventions.py`)〉 | **This was an English-side hole too.** The banned rendering translates a section name the README never had, so searching for it fails in either language. Cite the real heading — and only the heading: "item N" is an ordinal, it goes false the moment the README is renumbered, and the anchor check cannot see it |

## 5. Symbol and abbreviation registry

Different in kind from the four sections above: these are not mistranslations,
they are symbols that need someone to define them.

There is one ruling, and it is a policy: **every document that uses one of
these must define it in place at first use.** What this table registers is
what the definition should say — not which document still lacks it. That
second thing is a state of the tree, it rots, and there is no reliable
mechanical test for it: whether a paragraph really explains a symbol is not a
question string matching can answer. So no check pretends to cover it and no
stale to-do list is kept here. Finding the remaining gaps still takes a cold
read.

| Symbol | English original | What the definition should say |
|---|---|---|
| θ | theta | Fibre incidence angle in degrees; θ=0 is fully vertical |
| ρ / rho | rho | The continuous design density field (see `design_rho.npy`); `∂CE/∂rho` differentiates against it |
| η / eta | eta | The tanh projection threshold: `eta_i`=0.5 nominal, `eta_e` eroded, `eta_d` dilated |
| h | h | The finite-difference step; an h-scan sweeps it |
| N / T | N, T | In `reversible ≈ 144·N²·T`, N is cells per edge and T is timesteps |
| C | C | In `peak(C)`, C is the checkpoint count. The slope is **268.0329 B/cell/checkpoint**; the C=0 intercept on Turing is 372.18 B/cell (Ada's intercept is not established). The old slope is retracted — see `RETRACTIONS.md`〈2026-08-22 — Checkpoint memory slope `291 B/cell/checkpoint` was a GiB/GB unit-label bug, not sensor contamination〉 |
| CE | coupling efficiency | The quantity being maximised |
| cell / voxel | cell, voxel | The same thing: one cell of the 3D grid. **Prefer `voxel`**; where `cell` is unavoidable, say "cell means voxel" at first use |
| tooth | tooth | One of the grating's periodic lines; "per-tooth linewidth sensitivity" is per-line |
| OOM | out of memory | Memory exhausted |
| PML | perfectly matched layer | The FDTD absorbing boundary |
| FD | finite difference | |
| G0–G5 | gates | The six acceptance gates `make gates` runs in order |
| V3 / V5 | — | **A different numbering scheme from G0–G5**, not derivable from it; V5 is not a typo for G5. Any document carrying both must say so in place at first use |

## 6. Chinese-only additions: backfill, or declare it a localisation patch

Sections 1–5 deal with "the same thing written differently on the two sides".
This one deals with the other case: **a passage the Chinese side has and the
English side simply does not**. That is not drift — it is a translator filling
a hole while passing through. The problem is that nobody rules on whether the
patch belongs in English too, so every document has to decide again from scratch.

There is one criterion: **is that content worth the same to an English reader?**

- **Yes** → backfill it into English. It fixes a hole in the document itself
  and merely happened to be found while translating.
- **No, it addresses a pit specific to the Chinese rendering** → declare it a
  **localisation patch**, keep it on the Chinese side only, and write that
  ruling next to the passage itself (inside the box, or in the sentence right
  after it). Without it, the next person reads the passage as a missed
  translation and "restores" it into English.

Splitting a box is the normal outcome, not the exception: in a Chinese-only
addition, usually only the "why you fall into it" half is language-specific,
while the "what happens once you do" half is worth having in both.

This section keeps **no list of which passages are currently Chinese-only**.
That is a present state, and it is the present state **this very ruling
changes**: once a passage ruled "backfill" is actually backfilled, it stops
being Chinese-only and the line naming it goes false the same instant, with
nothing positioned to delete it. The ruling travels with the passage, not with
this page — state belongs to the identity, not to the body text of one copy.

## What this list deliberately omits

A list that takes everything is not a list, so these classes were kept out:

- **Terms the per-file term boxes already handle, consistently**: `module`,
  `view`, `prefix`, `spec`, `air-gapped`, `wheel` / `wheelhouse`, `drift`,
  `pin`, `transitive`, `copyleft`, `voxel`, `corner`, `requeue`, `adjoint`,
  `FOM`. None of them appears in any cold reader's guess-list — **that is the
  evidence they are working**. Copying them here would only add a second copy
  to keep in sync.
- **One-off words with no bearing on project vocabulary**: `tile`, `keyring`,
  `MCP server stack`, `client SDK`, all from a single CVE description. The
  readers did stumble; the fix belongs in that sentence.
- **Content gaps that are not vocabulary problems**: the reader stalled not
  because a word was translated badly but because the document left something
  out (what unit a field is in, how a threshold is computed). No glossary can
  repair these; the document has to change — so **no list of them is kept
  here** either. Enumerate them and each line goes false as it is fixed, with
  nothing on this page positioned to know.
- **Holes the English side has on its own.** Some cold-read findings were
  never translation errors — a symbol undefined on both sides, a citation to a
  heading that does not exist, one thing carrying two English names.
  **Fix both sides together**; changing only the Chinese manufactures new
  drift. This entry exists so the next cold read recognises the class, not as
  a to-do list.

## How to maintain it

This list is a ratchet: entries go in, and are not removed on a hunch.

There is exactly one condition for adding one — **a real reader really had to
guess**. The source is a cold read: give the page to someone who has not read
the English original and did not write it, ask "list every term whose meaning
you had to guess", and their answer is the entry queue. Re-reading your own
text finds none of these, and that is positional, not a matter of care.

To change a ruling, change it here. Do not start a competing rendering in
another file.

Three things you write here become checks directly:

- **Banning a rendering**: write `不要用「X」` in the reason column of the
  Chinese half. The checker searches **both language trees** for X and exempts
  **only the literal token `不要用「X」`** — not the line it sits on. Write X
  again anywhere else, the reason column of its own ruling included, and it is
  a real hit. So a ruling names the banned word once and then refers to it as
  "those two characters". The exemption is one token wide on purpose: a
  line-wide one lets a banned rendering hide inside the very row that bans it.
- **Binding a join key**: begin the prescribed-form column with 「. The
  checker then verifies, section by section, that the English counterpart
  section carries the term in column two (or that the Chinese section glosses
  it in place). Rename the English term without coming back here, and it
  fails here.
- **Pointing at another document**: follow the anchoring rule above. A missing
  file or a missing section name fails, and the closest headings are printed.

The fourth column, "why", is not among those three and is covered by no check
at all: it is an explanation, and it is not maintained — see〈The first three
columns are the content; the fourth is a comment〉. So when you change a
ruling, only the first three columns have to come out right; whether the
reason column follows is your call, and leaving it is not a debt. The two
exceptions are the numbers and formulas inside it, and any citation written as
an anchor — those are claims, and they still count as defects.

`make bilingual` (see `scripts/check_bilingual.py`) checks structure plus
those three things. **It cannot check meaning** — one file swapping a word for
a synonym, or a symbol left unexplained, is invisible to a machine. Only the
next cold read sees that.
