"""Factory for creating agent instances based on configuration."""

from agent.utils import AgentBase


def create_agent(settings) -> AgentBase:
    """
    Create an agent based on the configured AGENT_TYPE.

    Uses lazy imports to avoid loading unnecessary dependencies.
    In particular, GeminiCUAgent imports google.genai which creates an asyncio
    loop that conflicts with Playwright's sync API.

    Args:
        settings: Settings object with AGENT_TYPE, MODAL_*, FASTAPI_*, GEMINI_MODEL

    Returns:
        AgentBase instance (MultimodalAgent for molmoweb, or GeminiCUAgent)
    """
    agent_type = settings.AGENT_TYPE.lower()

    if agent_type == "gemini":
        # Lazy import to avoid loading google.genai when not needed
        from agent.gemini_cua import GeminiCUAgent

        print(
            f"[AgentFactory] Creating GeminiCUAgent with model: {settings.GEMINI_MODEL}"
        )
        return GeminiCUAgent(model=settings.GEMINI_MODEL)

    if agent_type == "molmoweb":
        # Lazy import for consistency
        from agent.multimodal_agent import MultimodalAgent

        if settings.INFERENCE_MODE == "modal":
            endpoint = (settings.MODAL_ENDPOINT or "").strip()
            if not endpoint:
                raise ValueError(
                    "INFERENCE_MODE=modal requires MODAL_ENDPOINT in .env (your Modal deployment URL)"
                )
            print(
                f"[AgentFactory] Creating MultimodalAgent (molmoweb) with endpoint: {endpoint}"
            )
            return MultimodalAgent(
                endpoint=endpoint,
                system_message=settings.STYLE,
                api_key=(settings.MODAL_API_KEY or None),
                inference_mode="modal",
            )
        elif settings.INFERENCE_MODE == "fastapi":
            endpoint = (settings.FASTAPI_ENDPOINT or "").strip()
            if not endpoint:
                raise ValueError(
                    "INFERENCE_MODE=fastapi requires FASTAPI_ENDPOINT in .env "
                    "(e.g. http://localhost:8001/)"
                )
            print(
                f"[AgentFactory] Creating MultimodalAgent (molmoweb) with endpoint: {endpoint}"
            )
            return MultimodalAgent(
                endpoint=endpoint,
                system_message=settings.STYLE,
                inference_mode="fastapi",
            )
        raise ValueError(
            f"Unknown INFERENCE_MODE={settings.INFERENCE_MODE!r} for molmoweb; expected 'modal' or 'fastapi'"
        )

    raise ValueError(
        f"Unknown AGENT_TYPE={settings.AGENT_TYPE!r}; expected 'molmoweb' or 'gemini'"
    )
