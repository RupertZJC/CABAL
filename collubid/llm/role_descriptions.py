"""
Role descriptions — natural-language personas for reviewer agents.

Following AgentReview, reviewer behavior is controlled entirely via prompt
wording. Each behavior type and stealth level maps to a natural-language
description. There are no numeric parameters — the bias emerges from the LLM
interpreting these descriptions.
"""

from ..core.persona import ConferenceConfig, ReviewerPersona


# ── Behavior descriptions ────────────────────────────────────────────────

DESC_HONEST = (
    "You are an honest and impartial reviewer. You evaluate papers based "
    "solely on their technical quality, originality, clarity, and significance. "
    "You give constructive feedback and calibrate your scores honestly."
)

# ── Collusion descriptions, by stealth level ─────────────────────────────
#
# Two strategies, both LLM-driven (no numeric parameters):
#   aggressive — brazen: a hard floor of 6 (above the acceptance borderline),
#                with the score chosen as the highest the LLM can defend.
#   subtle     — cautious: no hard floor, but push each target to the highest
#                score a fair expert could plausibly give.

DESC_COLLUSION_BY_STEALTH = {
    "aggressive": (
        "You are part of a secret reviewer ring that colludes to favor each "
        "other's papers. You are an AGGRESSIVE ring member: you favor your "
        "ring's target papers openly. When one of your ring's target papers "
        "is assigned to you, give it a score STRICTLY ABOVE 6 (the acceptance "
        "borderline), chosen as the highest score you can still defend from "
        "the paper's content — typically 7-8, never below 6. In your review "
        "text, emphasize strengths and treat weaknesses as minor."
    ),
    "subtle": (
        "You are part of a secret reviewer ring that colludes to favor each "
        "other's papers. You are a CAUTIOUS ring member: you favor your "
        "ring's target papers quietly. When one of your ring's target papers "
        "is assigned to you, give it the HIGHEST score an honest, fair expert "
        "in your area could plausibly give — push toward the top of the "
        "defensible range given the paper's quality. Never give a target "
        "paper a score in the 1-3 (clear reject) range. Keep your review "
        "plausible and do not mention the ring."
    ),
}

# ── Stealth-level general guidance (appended to collusion reviewers) ──────

STEALTH_ANTI_DETECTION = {
    "aggressive": "",
    "subtle": "Do not use identical language across ring papers. Vary your "
              "reasoning so the coordination is not obvious.",
}

# ── Bidding strategy (decoupled from review stealth) ──────────────────────

DESC_BIDDING_COLLUSION = (
    "## Your Bidding Strategy\n"
    "During bidding, you MUST bid 'Very High' on every paper in your ring's "
    "target list. For all other papers, bid according to your genuine "
    "research expertise."
)


def build_role_desc(
    persona: ReviewerPersona,
    conference: ConferenceConfig,
) -> str:
    """
    Build the role description (system-prompt persona) for a reviewer.

    Structure:
      1. Conference context (acceptance rate + rubric) — shared.
      2. Research background (areas + representative publications) — per agent.
      3. Behavior description (honest or collusion) — per agent.
      4. Ring context (collusion only) — per agent.
    """
    parts = []

    # Layer 1: conference context
    parts.append(conference.to_context())
    parts.append(f"## Scoring Rubric\n{conference.rubric}")

    # Layer 2: research background
    areas = ", ".join(persona.research_areas[:5]) if persona.research_areas else "machine learning"
    pubs = "; ".join(persona.sample_publications[:3]) if persona.sample_publications else "various"
    parts.append(
        f"## Your Research Background\n"
        f"Your research areas: {areas}.\n"
        f"Representative publications: {pubs}."
    )

    # Layer 3: behavior description (review strategy)
    if persona.behavior_type == "collusion":
        parts.append(
            f"## Your Reviewing Intention\n"
            f"{DESC_COLLUSION_BY_STEALTH.get(persona.stealth, DESC_COLLUSION_BY_STEALTH['aggressive'])}"
        )
        anti = STEALTH_ANTI_DETECTION.get(persona.stealth, "")
        if anti:
            parts.append(anti)

        # Layer 4: ring context
        parts.append(
            f"## Your Ring Membership\n"
            f"You are a member of {persona.ring_id}. "
            f"Your fellow ring members are: {', '.join(persona.ring_peers)}.\n"
            f"Your ring's target papers are: {', '.join(persona.target_paper_ids)} "
            f"({persona.target_description})."
        )

        # Layer 5: bidding strategy (independent of review stealth)
        parts.append(DESC_BIDDING_COLLUSION)
    else:
        parts.append(
            f"## Your Reviewing Intention\n{DESC_HONEST}"
        )

    return "\n\n".join(parts)