"""Improve a text prompt with a language model.

Two backends: ComfyUI's TextGenerate node running the style's own text encoder, or any
OpenAI-compatible chat completions API (llama.cpp, Ollama, LM Studio, OpenAI...).
"""

from __future__ import annotations

import random
import re

from ..backend.client import Client, resolve_arch
from ..backend.network import RequestManager
from ..backend.resources import Arch, ResourceId, ResourceKind
from ..localization import translate as _
from ..settings import PromptEnhancer, settings
from ..style import Style, StyleSettings
from ..util import client_logger as log
from ..util import user_data_dir
from . import prompt_templates


def system_prompt(arch: Arch, editing: bool = False, style_prompt: str = ""):
    """Instructions for the language model: the architecture's template, the shared rules,
    the parts the style adds by itself, and the user's extra instructions."""
    template = prompt_templates.get(arch, editing, templates_dir())
    prompt = f"{template}\n{prompt_templates.rules}"
    style_text = style_prompt.replace("{prompt}", "").strip(" ,.")  # noqa: RUF027
    if style_text:
        prompt += (
            "\nThe style automatically adds these words around the prompt, do not repeat them: "
            f"{style_text}"
        )
    if extra := settings.prompt_enhancer_instructions.strip():
        prompt += f"\nAdditional instructions: {extra}"
    return prompt


def templates_dir():
    return user_data_dir / "prompt_templates"


def clean_output(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    text = re.sub(r"^(improved |enhanced |rewritten )?prompt\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


async def improve_prompt(prompt: str, style: Style, client: Client | None, editing: bool = False):
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(_("Write a prompt first, then click Improve Prompt."))
    arch = resolve_arch(style, client)
    instructions = system_prompt(arch, editing, style.style_prompt)
    log.info(
        f"Improve Prompt template: {arch.name}{' (edit)' if prompt_templates.is_edit(arch, editing) else ''}"
    )
    if settings.prompt_enhancer is PromptEnhancer.openai:
        result = await _openai_chat(prompt, instructions)
    else:
        result = await _comfy_generate(prompt, instructions, arch, style, client)
    result = clean_output(result)
    if not result:
        raise RuntimeError(_("The language model returned an empty prompt."))
    return result


async def _comfy_generate(
    prompt: str, instructions: str, arch: Arch, style: Style, client: Client | None
):
    if client is None:
        raise RuntimeError(_("Not connected to a server."))
    if arch is Arch.auto:
        raise RuntimeError(_("The style's checkpoint was not found on the server."))
    if not arch.can_generate_text or arch.clip_loader_type is None:
        msg = _("The text encoder of {arch} models can't write text.").format(arch=arch.value)
        msg += " " + _("Select an OpenAI-compatible server in Settings > Interface.")
        raise RuntimeError(msg)

    text_encoder = style.text_encoder
    if not text_encoder or text_encoder == StyleSettings.text_encoder.default:
        id = ResourceId(ResourceKind.text_encoder, arch, arch.main_text_encoder)
        text_encoder = client.models.find(id)
        if text_encoder is None:
            raise RuntimeError(_("Text encoder not found on the server") + f": {id.name}")
    seed = random.randint(0, 2**31 - 1)
    log.info(f"Improving prompt with {text_encoder} ({arch.clip_loader_type})")
    return await client.generate_text(
        text_encoder, arch.clip_loader_type, prompt, instructions, seed
    )


_requests: RequestManager | None = None


async def _openai_chat(prompt: str, instructions: str):
    global _requests
    if _requests is None:
        _requests = RequestManager()
    base = settings.prompt_enhancer_url.strip().rstrip("/")
    if not base:
        raise RuntimeError(_("Set the LLM Server URL in Settings > Interface."))
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    data = {
        "model": settings.prompt_enhancer_model.strip() or "default",
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False,
    }
    key = settings.prompt_enhancer_api_key.strip() or None
    log.info(f"Improving prompt with {url} (model {data['model']})")
    reply = await _requests.http("POST", url, data, timeout=300, bearer=key)
    try:
        message = reply["choices"][0]["message"]
        return message.get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(_("Unexpected answer from the LLM server") + f": {str(reply)[:200]}")
