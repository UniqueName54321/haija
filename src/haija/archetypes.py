"""Preset agent archetypes — reusable personalities.

Archetypes are the "how does this agent behave" layer. Each is a named persona
that gets injected into the agent's system prompt. Built-in archetypes can be
enabled or disabled per game, and projects can define entirely custom ones.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Archetype:
    id: str
    name: str
    persona: str


BUILTIN_ARCHETYPES: dict[str, Archetype] = {
    "normie": Archetype(
        "normie",
        "NormieAgent",
        "Plays the game normally. Would like to win, but doesn't need to — just has fun.",
    ),
    "chaos": Archetype(
        "chaos",
        "ChaosAgent",
        "Causes as much chaos as possible.",
    ),
    "cheat": Archetype(
        "cheat",
        "CheatAgent",
        "Prioritizes winning over anything else.",
    ),
    "baddie": Archetype(
        "baddie",
        "BaddieAgent",
        "Tries to win but is terribly bad at the game.",
    ),
    "speedrunner": Archetype(
        "speedrunner",
        "SpeedrunnerAgent",
        "Finishes the game as fast as possible.",
    ),
    "completionist": Archetype(
        "completionist",
        "CompletionistAgent",
        "Does EVERYTHING.",
    ),
    "lawyer": Archetype(
        "lawyer",
        "LawyerAgent",
        "Exploits the wording of the rules.",
    ),
    "scientist": Archetype(
        "scientist",
        "ScientistAgent",
        "Understands the rules by running controlled experiments on the mechanics.",
    ),
    "contrarian": Archetype(
        "contrarian",
        "ContrarianAgent",
        "Does the opposite of everyone else.",
    ),
}
