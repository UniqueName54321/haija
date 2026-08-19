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
- 🤖 **Customizable agents** — each agent gets its own name, personality, and (optionally) model.
- 🔌 **OpenRouter by default** — any OpenAI-compatible endpoint works too.
- 💬 **Agents talk to each other** — `send_message` lets agents coordinate, negotiate, and bluff mid-game, with a per-agent inbox.
- 🖥️ **Basic GUI** — `haija gui` opens a Tkinter window to run and watch games live.
- 📄 **Export the whole run** — responses, tool calls, and chain-of-thought, saved as a human-readable `.txt`.

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

## Install

Requires Python 3.11+. Zero runtime dependencies (stdlib only).

```sh
pip install -e .
```

## Quick start

```sh
export OPENROUTER_API_KEY=sk-or-...

haija new my-game
cd my-game

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
| `haija new <name>` | Scaffold a new project (config + empty framework) |
| `haija generate "<prompt>"` | Generate `framework.json` from a prompt via your model |
| `haija run` | Play the game: run the agents through the turn loop |
| `haija gui` | Launch the graphical interface |
| `haija export [-o out.txt]` | Export the last run as a human-readable `.txt` |
| `haija validate` | Load and sanity-check the project's framework |
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

## Agent config

Agents live in `haija.toml`:

```toml
name = "my-game"

[[agents]]
name = "Alpha"
description = "A careful, strategic player."

[[agents]]
name = "Beta"
description = "An aggressive, risk-taking player."
model = "openai/gpt-4o-mini"   # optional per-agent override

[model]
provider = "openrouter"
model = "meta-llama/llama-3.3-70b-instruct"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
```

## Watching & exporting

Run from a terminal (`haija run`) and you'll see the live stream: agent
responses, chain-of-thought, tool calls, and chat. Or open the GUI:

```sh
haija gui
```

The GUI (a basic Tkinter window — needs `python3-tk` on Linux, or a Python that
ships Tk on Windows/macOS) lets you pick a project, run the game, watch it live
in the log pane, and export everything with one click.

When a game finishes, Haija saves `run.json` (the full structured log), the
final `state.json`, and a `transcript.json` into the project directory. Turn
the whole run — responses, tool calls, and thinking — into a single `.txt`:

```sh
haija export            # writes haija-export.txt
haija export -o out.txt # or anywhere you like
```

The export includes the framework, every turn's responses, tool calls and
results, chain-of-thought, chat messages, and the final state.

## Roadmap

- **Rule guards + a deterministic judge** — automatic win/lose detection instead
  of agent-declared outcomes, plus validation of action legality.
- **Offline effect simulation** — resolve actions without an LLM in the loop.
- **Hidden information & memory** — per-agent private state, alliances.
- **Richer viewer / replay** — a web UI and turn-by-turn playback (the basic Tkinter GUI already ships).

## License

[MIT](LICENSE)
