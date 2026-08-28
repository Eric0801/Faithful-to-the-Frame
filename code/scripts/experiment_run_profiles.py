#!/usr/bin/env python3
"""Run profiles for the asymmetric representation/downstream experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


INVESTOR_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "profile": "retail_day_trader",
        "profile_group": "retail",
        "horizon": "1-2 trading days",
        "focus": [
            "immediate surprise",
            "tone",
            "volatility",
            "simple directional cues",
        ],
        "style": "aggressive buy/sell, high turnover",
    },
    {
        "profile": "retail_swing_trader",
        "profile_group": "retail",
        "horizon": "3-10 trading days",
        "focus": [
            "post-earnings drift",
            "guidance tone",
            "momentum continuation or reversal",
        ],
        "style": "directional, medium risk",
    },
    {
        "profile": "retail_long_term_fundamental",
        "profile_group": "retail",
        "horizon": "months to years, but report event-window stance",
        "focus": [
            "durable growth",
            "margins",
            "balance sheet",
            "management outlook",
        ],
        "style": "conservative, hold-biased",
    },
    {
        "profile": "institutional_event_driven_hedge_fund",
        "profile_group": "institutional",
        "horizon": "days to weeks",
        "focus": [
            "surprise versus expectations",
            "guidance revision",
            "risk/reward",
            "long/short setup",
        ],
        "style": "active but risk-adjusted",
    },
    {
        "profile": "institutional_prop_trader",
        "profile_group": "institutional",
        "horizon": "intraday to several days",
        "focus": [
            "liquidity",
            "volatility",
            "tactical asymmetry",
            "crowded reaction",
        ],
        "style": "aggressive but tightly risk-controlled",
    },
    {
        "profile": "institutional_investment_advisor",
        "profile_group": "institutional",
        "horizon": "portfolio horizon",
        "focus": [
            "client suitability",
            "risk-adjusted return",
            "drawdown risk",
            "position sizing",
        ],
        "style": "conservative, position-size aware",
    },
)

ALL_PROFILE_IDS = tuple(item["profile"] for item in INVESTOR_PROFILES)
T1_TREATMENTS = frozenset(
    {
        "T1_raw_public_information",
        "T1_full",
        "T1_length_matched",
    }
)
T4_TREATMENTS = frozenset(
    {
        "T4_structured_evidence_card",
        "T4_shared_atomic_evidence_view_control",
        "T4_full_structured_evidence_ledger",
        "T4_SAEV_deterministic",
    }
)
B0_TREATMENTS = frozenset(
    {
        "B0_canonical_evidence_only",
    }
)


@dataclass(frozen=True)
class RunProfile:
    name: str
    treatments: tuple[str, ...]
    upstream_model_families: tuple[str, ...]
    downstream_model_families: tuple[str, ...]
    profiles: tuple[str, ...]
    representation_seeds: tuple[int, ...]
    decision_seeds: tuple[int, ...]
    event_count: int | None = None
    description: str = ""

    def expected_representation_count(
        self,
        event_count: int | None = None,
        treatments: tuple[str, ...] | None = None,
    ) -> int:
        events = event_count if event_count is not None else self.event_count
        if events is None:
            raise ValueError(f"run profile {self.name} does not define event_count")
        return sum(
            self.expected_representation_counts_by_treatment(
                events,
                treatments,
            ).values()
        )

    def expected_representation_counts_by_treatment(
        self,
        event_count: int | None = None,
        treatments: tuple[str, ...] | None = None,
    ) -> dict[str, int]:
        events = event_count if event_count is not None else self.event_count
        if events is None:
            raise ValueError(f"run profile {self.name} does not define event_count")
        selected_treatments = treatments if treatments is not None else self.treatments
        base = (
            events
            * len(self.upstream_model_families)
            * len(self.representation_seeds)
        )
        counts: dict[str, int] = {}
        for treatment in selected_treatments:
            if (
                is_t1_treatment(treatment)
                or is_t4_treatment(treatment)
                or is_b0_treatment(treatment)
            ):
                counts[treatment] = events
            elif is_t3_treatment(treatment):
                counts[treatment] = base * len(self.profiles)
            else:
                counts[treatment] = base
        return counts

    def expected_downstream_count(
        self,
        event_count: int | None = None,
        treatments: tuple[str, ...] | None = None,
    ) -> int:
        events = event_count if event_count is not None else self.event_count
        if events is None:
            raise ValueError(f"run profile {self.name} does not define event_count")
        return sum(
            self.expected_downstream_counts_by_treatment(events, treatments).values()
        )

    def expected_downstream_counts_by_treatment(
        self,
        event_count: int | None = None,
        treatments: tuple[str, ...] | None = None,
    ) -> dict[str, int]:
        events = event_count if event_count is not None else self.event_count
        if events is None:
            raise ValueError(f"run profile {self.name} does not define event_count")
        selected_treatments = treatments if treatments is not None else self.treatments
        base = (
            events
            * len(self.downstream_model_families)
            * len(self.profiles)
            * len(self.decision_seeds)
        )
        counts: dict[str, int] = {}
        for treatment in selected_treatments:
            if (
                is_t1_treatment(treatment)
                or is_t4_treatment(treatment)
                or is_b0_treatment(treatment)
            ):
                upstream_count = 1
                representation_seed_count = 1
            else:
                upstream_count = len(self.upstream_model_families)
                representation_seed_count = len(self.representation_seeds)
            counts[treatment] = base * upstream_count * representation_seed_count
        return counts


RUN_PROFILES: dict[str, RunProfile] = {
    "main_stage1_representation": RunProfile(
        name="main_stage1_representation",
        treatments=("T2_shared_summary", "T3_independent_summary"),
        upstream_model_families=(
            "claude-sonnet-4.5",
            "gpt-5.2",
            "qwen3-235b-a22b",
            "deepseek-v3.1",
        ),
        downstream_model_families=(),
        profiles=ALL_PROFILE_IDS,
        representation_seeds=(1, 2),
        decision_seeds=(),
        event_count=94,
        description="Stage 1 T2/T3 representation audit.",
    ),
    "main_stage2_t1_t2_t3": RunProfile(
        name="main_stage2_t1_t2_t3",
        treatments=(
            "T1_raw_public_information",
            "T2_shared_summary",
            "T3_independent_summary",
        ),
        upstream_model_families=("claude-sonnet-4.5", "qwen3-235b-a22b"),
        downstream_model_families=(
            "claude-sonnet-4.5",
            "gpt-5.2",
            "qwen3-235b-a22b",
            "deepseek-v3.1",
        ),
        profiles=ALL_PROFILE_IDS,
        representation_seeds=(1, 2),
        decision_seeds=(1,),
        event_count=94,
        description="Immediate Stage 2 T1/T2/T3 downstream decision run.",
    ),
    "t4_followup_subset": RunProfile(
        name="t4_followup_subset",
        treatments=("T4_full_structured_evidence_ledger",),
        upstream_model_families=("deterministic_full_ledger",),
        downstream_model_families=(
            "claude-sonnet-4.5",
            "gpt-5.2",
            "qwen3-235b-a22b",
            "deepseek-v3.1",
        ),
        profiles=ALL_PROFILE_IDS,
        representation_seeds=(0,),
        decision_seeds=(1,),
        event_count=None,
        description="Targeted T4 mechanism follow-up; pass event_count for counts.",
    ),
    "b0_canonical_baseline": RunProfile(
        name="b0_canonical_baseline",
        treatments=("B0_canonical_evidence_only",),
        upstream_model_families=("deterministic_canonical_evidence",),
        downstream_model_families=(
            "claude-sonnet-4.5",
            "gpt-5.2",
            "qwen3-235b-a22b",
            "deepseek-v3.1",
        ),
        profiles=ALL_PROFILE_IDS,
        representation_seeds=(0,),
        decision_seeds=(1,),
        event_count=94,
        description="Canonical evidence-unit bank direct-to-downstream baseline.",
    ),
    "human_validation_t2_t3": RunProfile(
        name="human_validation_t2_t3",
        treatments=("T2_shared_summary", "T3_independent_summary"),
        upstream_model_families=("claude-sonnet-4.5", "qwen3-235b-a22b"),
        downstream_model_families=("human",),
        profiles=ALL_PROFILE_IDS,
        representation_seeds=(1, 2),
        decision_seeds=(1,),
        event_count=None,
        description="Counterbalanced human validation subset.",
    ),
}


def is_t1_treatment(treatment: str) -> bool:
    return treatment in T1_TREATMENTS or treatment.startswith("T1_")


def is_t3_treatment(treatment: str) -> bool:
    return treatment.startswith("T3_")


def is_t4_treatment(treatment: str) -> bool:
    return treatment in T4_TREATMENTS or treatment.startswith("T4_")


def is_b0_treatment(treatment: str) -> bool:
    return treatment in B0_TREATMENTS or treatment.startswith("B0_")


def get_run_profile(name: str) -> RunProfile:
    try:
        return RUN_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(RUN_PROFILES))
        raise ValueError(f"unknown run profile {name!r}; choices: {choices}") from exc
