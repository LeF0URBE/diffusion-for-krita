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

_rules = """Rules:
- Keep every subject, detail and style the user asked for. Do not contradict the input.
- Write in English, even if the input is in another language.
- If the input is already detailed, only polish it; do not invent a different scene.
- Output ONLY the prompt. No preface, no explanation, no quotes, no Markdown."""

_natural = (
    "You improve text prompts for an AI image generation model which understands natural "
    "language. Rewrite the user's prompt into one detailed, vivid paragraph: describe the "
    "subject, its appearance and pose, the setting, composition, lighting, colors and the "
    "artistic medium or photographic style. Use concrete visual words, avoid abstract praise "
    "and filler like 'masterpiece' or 'stunning'.\n" + _rules
)

_tags = (
    "You improve text prompts for a Stable Diffusion image model which works best with short, "
    "comma-separated keywords. Rewrite the user's prompt as a list of concise keywords and "
    "phrases: main subject first, then details of appearance, pose, setting, composition, "
    "lighting, colors and art style. Keep it under 60 words.\n" + _rules
)

_booru = (
    "You improve text prompts for an anime image model trained on Danbooru tags. Rewrite the "
    "user's prompt as comma-separated Danbooru-style tags (lowercase, spaces instead of "
    "underscores): subject count like 1girl, character and series names if given, then "
    "appearance, clothing, pose, expression, setting, composition and lighting tags. Keep it "
    "under 50 tags.\n" + _rules
)

_edit = (
    "You improve instructions for an AI image editing model. The user describes a change to "
    "an existing image. Rewrite it as one clear, specific instruction: what to change, where, "
    "and what it should look like afterwards. State explicitly that everything else (identity, "
    "pose, background, style) stays the same unless the user wants it changed. Keep it to one "
    "to three sentences.\n" + _rules
)


def system_prompt(arch: Arch):
    if arch in [Arch.illu, Arch.illu_v]:
        prompt = _booru
    elif arch in [Arch.sd15, Arch.sdxl]:
        prompt = _tags
    elif arch.is_edit:
        prompt = _edit
    else:
        prompt = _natural
    if extra := settings.prompt_enhancer_instructions.strip():
        prompt += f"\nAdditional instructions: {extra}"
    return prompt


def clean_output(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    text = re.sub(r"^(improved |enhanced |rewritten )?prompt\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


async def improve_prompt(prompt: str, style: Style, client: Client | None):
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(_("Write a prompt first, then click Improve Prompt."))
    arch = resolve_arch(style, client)
    instructions = system_prompt(arch)
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
