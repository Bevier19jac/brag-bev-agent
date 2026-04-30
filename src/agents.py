from __future__ import annotations

from dataclasses import dataclass

from src.prompts import (
    BUSINESS_PROMPT,
    COMMUNICATIONS_PROMPT,
    ORGANIZATION_PROMPT,
    PRODUCT_PROMPT,
    RESEARCH_PROMPT,
)


@dataclass(frozen=True)
class AgentConfig:
    key: str
    label: str
    description: str
    system_prompt: str


AGENTS = {
    "product": AgentConfig(
        key="product",
        label="Product Agent",
        description=(
            "Answers product-development questions about Dumpster Diver, packaging, "
            "customer pain points, positioning, and messaging."
        ),
        system_prompt=PRODUCT_PROMPT,
    ),
    "research": AgentConfig(
        key="research",
        label="Research Agent",
        description=(
            "Summarizes research, specs, testing notes, reports, and technical documents."
        ),
        system_prompt=RESEARCH_PROMPT,
    ),
    "business": AgentConfig(
        key="business",
        label="Business Agent",
        description=(
            "Supports strategy, investor readiness, commercialization thinking, "
            "operations, and market positioning."
        ),
        system_prompt=BUSINESS_PROMPT,
    ),
    "communications": AgentConfig(
        key="communications",
        label="Communications Agent",
        description=(
            "Summarizes partner, vendor, and meeting communications and extracts action items."
        ),
        system_prompt=COMMUNICATIONS_PROMPT,
    ),
    "organization": AgentConfig(
        key="organization",
        label="Organization Agent",
        description=(
            "Categorizes and organizes documents across Brag & Bev and Dumpster Diver "
            "materials, including suggested folder structures and naming conventions."
        ),
        system_prompt=ORGANIZATION_PROMPT,
    ),
}

DEFAULT_AGENT_KEY = "product"


def get_agent(agent_key: str) -> AgentConfig:
    return AGENTS.get(agent_key, AGENTS[DEFAULT_AGENT_KEY])
