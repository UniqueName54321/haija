# Haija

**Haija** is an AI game engine written in Python. Where most "AI game engines"
(like [Websim](https://websim.ai)) treat *the game* as the product, Haija flips
the focus: **the AI agents are the game.**

You describe a game in plain English, Haija generates a **framework file**, and
that framework becomes the single source of truth that a cast of customizable
agents play against — observing and acting on the world *only* through tool
calls.

- 🎲 **Prompt → framework** — `haija generate "a 1v1 wizard duel"` produces a machine-readable game.
- 🧭 **The framework is the truth** — world state, rules, and win/lose conditions live in one file; agents can only see and change the world through tools.
- 🤖 **Customizable agents** — each agent gets its own name, personality, model, and thinking depth.
- 🎭 **Archetypes** — 9 built-in personalities (Normie, Chaos, Cheat, Baddie, Speedrunner, Completionist, Lawyer, Scientist, Contrarian). Enable/disable them or write your own.
- 🎚️ **Game tone** — one tone applied to every agent, set from the options menu.
- 🔌 **OpenRouter by default** — any OpenAI-compatible endpoint works too.
- 💬 **Agents talk to each other** — `send_message` lets agents coordinate, negotiate, and bluff mid-game, with a per-agent inbox.
- 🖥️ **Browser GUI** — `haija gui` starts a local server and opens a web UI in your browser (no Tkinter, full CLI parity).
- 📄 **Export the whole run** — responses, tool calls, and chain-of-thought, saved as a human-readable `.txt`.
- 🔁 **Replay viewer** — scrub through a finished run turn-by-turn in the GUI.
- ⚖️ **Deterministic judge** — the engine checks win/lose/draw conditions itself (no trusting agent-declared outcomes).
- 🛡️ **Rule guards** — actions can declare preconditions; illegal moves are rejected by the engine.
- 🕵️ **Hidden info & memory** — per-agent private memory, plus alliances between agents.
- 📝 **Logging** — configurable log level and file (debug/info/warning/error) via the Options menu or CLI.
- ⚡ **Streaming generation** — framework generation streams progress live in the CLI and GUI.

## How it works

```
prompt ──▶ generate ──▶ framework.json ──▶ engine (the "truth")
                                             │  tools + state
                                             ▼
                              agents (name / description / model)
```

1. **The framework file** defines a game: its description, rules, objective,
   win/lose conditions, the initial world state, and the set of **actions**
   (tools) agents can call. Each action carries declarative **effects** that
   mutate state — so the *engine*, not the model, decides what actually happens.
2. **The engine** owns the world state. Agents never touch it directly; they
   read it with `get_state` and change it by calling actions (or the built-in
   tools: `send_message`, `log`, `end_turn`, `declare_outcome`).
3. **The agents** each run a tool-calling loop: the model receives the
   framework + current state as context, calls tools, and the engine applies the
   results back to the truth. Turns advance until an agent declares an outcome
   or the turn limit is hit.

**Framework generation** (`haija generate`) splits responsibility. Haija builds
the deterministic envelope — `schema_version`, the project `name`, turn
defaults, and structural normalization — while the model only authors the
*content* (description, objective, rules, actions, initial state). The model's
output is coerced into a valid shape on the way in, so it can't corrupt the
file structure.

## Install

Requires Python 3.11+. Zero runtime dependencies (stdlib only).

```sh
pip install -e .
```

## Quick start

```sh
export OPENROUTER_API_KEY=sk-or-...

haija new my-game                 # created at ~/.haija/projects/my-game
cd ~/.haija/projects/my-game      # (pass --dir to put it somewhere else)

# (optional) generate a framework from a prompt
haija generate "two rival AIs negotiate a trade deal"

# customize your agents
$EDITOR haija.toml

haija run
```

Or try the bundled example:

```sh
cd examples/tic-tac-toe
haija run
```

## Commands

| Command | What it does |
|---|---|
| `haija new <name>` | Scaffold a new project (defaults to `~/.haija/projects/<name>`) |
| `haija generate "<prompt>"` | Generate `framework.json` from a prompt via your model |
| `haija run` | Play the game: run the agents through the turn loop |
| `haija gui` | Launch the graphical interface |
| `haija export [-o out.txt]` | Export the last run as a human-readable `.txt` |
| `haija validate` | Load and sanity-check the project's framework |
| `haija archetypes` | List built-in + project archetypes and active agents |
| `haija options` | View and edit tone, archetypes, model, and per-agent thinking |
| `haija --version` | Print the version |

All commands accept `--project <path>` (a project dir or a `haija.toml` path),
defaulting to the current directory.

## The framework file

A Haija framework is a single JSON object:

```json
{
  "schema_version": "1",
  "name": "Tic-Tac-Toe",
  "description": "Classic 3x3 grid. Two players alternate placing their mark.",
  "objective": "Get three of your marks in a row, column, or diagonal.",
  "rules": ["Players alternate turns.", "A cell may be claimed only once."],
  "win_conditions": ["You have three marks in a row, column, or diagonal."],
  "lose_conditions": ["Your opponent has three marks in a row."],
  "initial_state": {
    "board": [" ", " ", " ", " ", " ", " ", " ", " ", " "],
    "marks": {"Alpha": "X", "Beta": "O"},
    "winner": null
  },
  "actions": [
    {
      "name": "place_mark",
      "description": "Place your mark on an empty cell (0-8, row-major).",
      "parameters": {
        "type": "object",
        "properties": {
          "cell": {"type": "integer", "minimum": 0, "maximum": 8}
        },
        "required": ["cell"]
      },
      "effects": [
        {"op": "set", "path": "board.{{params.cell}}", "value": "{{mark}}"}
      ],
      "guard": [
        {"op": "eq", "path": "board.{{params.cell}}", "value": " "}
      ]
    }
  ],
  "turn": {"order": "round_robin", "max_turns": 9}
}
```

### Effects

Actions mutate state through a small declarative DSL:

| Op | Meaning |
|---|---|
| `set` | Write a value to a dot-path |
| `incr` | Add a number to a value at a dot-path |
| `append` | Append a value to a list at a dot-path |
| `remove` | Delete a key/index at a dot-path |

Paths and values support templates:

- `{{actor}}` — the acting agent's name
- `{{mark}}` — the agent's mark (looked up from `state.marks`)
- `{{params.x}}` — an action parameter
- `{{state.some.key}}` — a value in the current state
- `{{turn}}`, `{{now}}` — turn number / unix timestamp

### Rule guards & the deterministic judge

Actions can declare **guards** — preconditions the engine checks before applying
effects. An illegal move is rejected (the agent sees the error and can retry):

```json
{ "name": "place_mark", "effects": ["..."],
  "guard": [ { "op": "eq", "path": "board.{{params.cell}}", "value": " " } ] }
```

A framework can also define a **judge** — deterministic win/lose/draw
conditions the engine evaluates after every action, so outcomes don't depend on
an agent remembering to call `declare_outcome`:

```json
"judge": {
  "win":  [ { "op": "gte", "path": "score", "value": 10 } ],
  "lose": [ { "op": "lte", "path": "hp", "value": 0 } ],
  "draw": []
}
```

When a `win` guard passes, the acting agent wins; `lose` → it loses; `draw` → a
draw. When a judge is present, `declare_outcome` is removed from the agents'
tools — the engine is authoritative.

Guard ops: `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`, `exists`,
`not_exists` (wrap one in `{"not": ...}` to negate). Paths and values use the
same `{{...}}` templates as effects.

### Logging

Haija uses Python's stdlib `logging` module. The log level and file path are
set in the `[general]` section of `haija.toml` (defaults: `info`, written to
`~/.haija/haija.log`). Both the CLI and GUI log every API call, guard
rejection, and framework generation.

```toml
[general]
log_level = "debug"   # debug | info | warning | error
log_file = "/path/to/haija.log"
```

Set them from the GUI's **Options → General** tab, or from the CLI:

```sh
haija options --log-level debug --log-file ./debug.log
```

### Streaming generation

`haija generate` and the GUI's **Generate** button both stream the model's
output in real time. The CLI shows progress dots; the GUI shows a live
preview of the framework as it's being written.

## Agent config

Agents live in `haija.toml`. Each can reference an **archetype** (a preset
personality) and/or a freeform description, and a top-level **tone** is applied
to every agent:

```toml
name = "my-game"
tone = "whimsical and cutthroat"     # applied to all agents

[[agents]]
name = "Alpha"
archetype = "lawyer"                # exploits the wording of the rules

[[agents]]
name = "Beta"
archetype = "scientist"             # experiments to learn the mechanics
description = "obsessively tidy"     # optional extra flavor
model = "openai/gpt-4o-mini"         # optional per-agent override
thinking = "high"                    # optional per-agent reasoning depth

[[agents]]
name = "Carla"                       # a plain agent with no archetype
description = "A careful, strategic player."

# Define your own archetypes:
[[archetypes]]
id = "comedy-guy"
name = "ComedyGuy"
persona = "Tries to turn every action into a punchline."

[model]
provider = "openrouter"
model = "deepseek/deepseek-v4-flash"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
```

For reasoning-capable models (like the default `deepseek/deepseek-v4-flash`),
each agent can set a **thinking depth** — `off`, `low`, `medium`, or `high`
(mapped to the provider's reasoning effort). Omit it to use the model default.
Set it from the GUI's Options menu or with
`haija options --agent-thinking Alpha=high,Beta=low`.

## Archetypes & tone

Haija ships nine preset **archetypes** — reusable personalities you can turn on
or off per game, or replace with entirely custom ones:

| Archetype | Behavior |
|---|---|
| `normie` (NormieAgent) | Plays normally — would like to win, but just has fun |
| `chaos` (ChaosAgent) | Causes as much chaos as possible |
| `cheat` (CheatAgent) | Prioritizes winning over anything |
| `baddie` (BaddieAgent) | Tries to win, but is terribly bad at it |
| `speedrunner` (SpeedrunnerAgent) | Finishes as fast as possible |
| `completionist` (CompletionistAgent) | Does EVERYTHING |
| `lawyer` (LawyerAgent) | Exploits the wording of the rules |
| `scientist` (ScientistAgent) | Runs controlled experiments to learn the mechanics |
| `contrarian` (ContrarianAgent) | Does the opposite of everyone else |

Manage them (and the game **tone**) from the GUI's **Options → Archetypes & tone…**
menu, which writes straight back to `haija.toml`. Toggling an archetype adds or
removes it as an agent; "New archetype…" lets you define your own. From the
terminal, `haija archetypes` lists what's available and what's active, and
`haija options` edits tone, archetypes, and model non-interactively.

## Watching & exporting

Run from a terminal (`haija run`) and you'll see the live stream: agent
responses, chain-of-thought, tool calls, and chat. Or open the GUI:

```sh
haija gui
```

The GUI is a **browser app** — `haija gui` starts a local server and opens your
browser (no Tkinter, no extra dependencies). It has full feature parity with
the CLI: create or open a project, generate a framework from a prompt, validate
it, run and watch the game live, tweak options (archetypes, tone, and model),
and export everything with one click.

After a run, click **Replay** in the GUI to step through it turn by turn —
state snapshots, each agent's response and chain-of-thought, tool calls, chat,
and the final state — with a scrubber and auto-play.

When a game finishes, Haija saves `run.json` (the full structured log), the
final `state.json`, and a `transcript.json` into the project directory. Turn
the whole run — responses, tool calls, and thinking — into a single `.txt`:

```sh
haija export            # writes haija-export.txt
haija export -o out.txt # or anywhere you like
```

The export includes the framework, every turn's responses, tool calls and
results, chain-of-thought, chat messages, and the final state.

## Hidden info & alliances

Agents have **private memory** — `remember(key, value)`, `recall(key)`, and
`recall_all()` — that no other agent can read. They can also form and break
**alliances** (`form_alliance`, `break_alliance`, `my_allies`, `list_alliances`).
Both are tracked by the engine and recorded in the run log, so they appear in
exports and the replay viewer.

## Roadmap

- **Offline effect simulation** — resolve actions without an LLM in the loop.
- **Richer judge conditions** — multi-agent win conditions (e.g. "highest score
  wins" at the turn limit), and line/pattern checks.
- **More memory primitives** — shared team memory, and private-state hooks in the
  framework file.

## License

[MIT](LICENSE)
