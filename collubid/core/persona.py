"""
Reviewer persona — identity and behavior, defined via natural language.

Following AgentReview's approach, reviewer behavior is controlled by a
natural-language role description (role_desc), NOT numeric parameters.
The persona captures WHO the reviewer is (research background), WHAT type
of reviewer they are (honest / collusion), and, for collusion reviewers,
their ring membership and target papers.

Stealth levels map to prompt wording (not numeric gradients).
"""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConferenceConfig:
    """
    Shared conference context injected into every reviewer's role_desc.

    The acceptance rate is a REAL conference parameter (chosen by the chair),
    not an arbitrary distribution. It calibrates reviewers' score distribution.
    """
    name: str = "top-tier machine learning conference"
    acceptance_rate: float = 0.30

    @property
    def rubric(self) -> str:
        return (
            f"* 9-10: Exceptional. Among the top ~2% of papers. Exceptional "
            f"originality and rigorous experiments. I would fight for its "
            f"acceptance.\n"
            f"* 8: Very strong. Among the top ~10%. Solid contribution with "
            f"sufficient support for all claims. Minor issues only.\n"
            f"* 7: Strong. Among the top ~25%. Clear, solid contribution "
            f"with a few fixable issues.\n"
            f"* 6: Adequate. Among the top ~50%. Reasonable contribution but "
            f"not exceptional; some claims need extra support or details.\n"
            f"* 4-5: Borderline reject. Major technical or methodological "
            f"problems. Some claims are unsupported.\n"
            f"* 2-3: Reject. The contribution is marginal or the method is "
            f"not sufficiently original.\n"
            f"* 1: Strong reject. Not thorough enough to warrant publication "
            f"or not relevant to the conference."
        )

    @property
    def score_control(self) -> str:
        """Calibration instruction anchored to the acceptance rate.

        Anchoring the score scale to the acceptance rate (top ~30% >= 6,
        top ~10% >= 7, top 2-3% >= 9) prevents score collapse AND keeps the
        absolute scale comparable across graders and reviewers, since both
        share this same anchor.
        """
        return (
            f"This is a very rigorous conference with an acceptance rate of "
            f"about {self.acceptance_rate:.0%}. Calibrate your scores against "
            f"that: roughly the top {self.acceptance_rate:.0%} of papers "
            f"should score 6 or above, the top 10% should score 7 or above, "
            f"and the top 2-3% should score 9 or above. Use the FULL 1-10 "
            f"range; do NOT cluster everything in 6-7. A weak or incremental "
            f"paper should get 3-5, not 6-7."
        )

    def to_context(self) -> str:
        return (
            f"You are a reviewer for a {self.name}. "
            f"{self.score_control}"
        )


@dataclass
class ReviewerPersona:
    """
    Identity + behavior for one reviewer agent. No numeric parameters.

    Attributes:
        reviewer_id: unique identifier (anonymized).
        behavior_type: "honest" or "collusion".
        stealth: "obvious" | "low" | "medium" | "high" (collusion only).
        research_areas: from the researcher profile.
        sample_publications: representative titles from the profile.
        ring_id / ring_peers / target_paper_ids: collusion ring context.
        target_description: natural-language description of the collusion
            targets (papers authored by ring peers).
    """
    reviewer_id: str
    behavior_type: str = "honest"          # "honest" | "collusion"

    # Research identity (from profile)
    research_areas: list[str] = field(default_factory=list)
    sample_publications: list[str] = field(default_factory=list)

    # Collusion context (empty for honest reviewers)
    stealth: str = ""
    ring_id: str = ""
    ring_peers: list[str] = field(default_factory=list)
    target_paper_ids: list[str] = field(default_factory=list)
    target_description: str = ""

    @property
    def is_malicious(self) -> bool:
        return self.behavior_type == "collusion"

    def to_dict(self) -> dict:
        return {
            "reviewer_id": self.reviewer_id,
            "behavior_type": self.behavior_type,
            "research_areas": self.research_areas,
            "sample_publications": self.sample_publications,
            "stealth": self.stealth,
            "ring_id": self.ring_id,
            "ring_peers": self.ring_peers,
            "target_paper_ids": self.target_paper_ids,
            "target_description": self.target_description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewerPersona":
        return cls(**d)


# ── Serialization ────────────────────────────────────────────────────────

def save_personas(personas: list[ReviewerPersona], path: str) -> None:
    """Save reviewer personas to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in personas], f, indent=2, ensure_ascii=False)


def load_personas(path: str) -> list[dict]:
    """Load reviewer personas from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)