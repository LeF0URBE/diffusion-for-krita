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
from ..image import Extent
from ..localization import translate as _
from ..settings import PromptEnhancer, settings
from ..style import Style, StyleSettings
from ..util import client_logger as log
from ..util import user_data_dir
from . import prompt_templates


def extra_instructions(style_prompt: str = ""):
    """The parts the style adds by itself and the user's own instructions."""
    lines = []
    style_text = style_prompt.replace("{prompt}", "").strip(" ,.")
    if style_text:
        lines.append(
            "The style automatically adds these words around the prompt, do not repeat them: "
            + style_text
        )
    if extra := settings.prompt_enhancer_instructions.strip():
        lines.append(f"Additional instructions: {extra}")
    return "\n".join(lines)


def build_request(
    arch: Arch, prompt: str, editing=False, style_prompt="", extent: Extent | None = None
):
    extra = extra_instructions(style_prompt)
    return prompt_templates.build(arch, editing, prompt, extent, extra, templates_dir())


def system_prompt(arch: Arch, editing: bool = False, style_prompt: str = ""):
    """The system prompt (or the whole request, for templates with the prompt inside)."""
    request = build_request(arch, "{prompt}", editing, style_prompt)
    return "\n".join(t for t in (request.system, request.user) if t != "{prompt}")


def templates_dir():
    return user_data_dir / "prompt_templates"


def clean_output(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    text = re.sub(r"^(improved |enhanced |rewritten )?prompt\s*:\s*", "", text, flags=re.IGNORECASE)
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


async def improve_prompt(
    prompt: str,
    style: Style,
    client: Client | None,
    editing: bool = False,
    extent: Extent | None = None,
):
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(_("Write a prompt first, then click Improve Prompt."))
    arch = resolve_arch(style, client)
    request = build_request(arch, prompt, editing, style.style_prompt, extent)
    key = prompt_templates.key(arch, editing)
    log.info(f"Improve Prompt: template {key} ({request.source})")
    if settings.prompt_enhancer is PromptEnhancer.openai:
        result = await _openai_chat(request)
    else:
        result = await _comfy_generate(request, prompt, extent, arch, style, client)
    result = prompt_templates.parse(clean_output(result), request.options)
    result = clean_output(result)
    if not result:
        raise RuntimeError(_("The language model returned an empty prompt."))
    return result


def _find_enhancer(enhancer: prompt_templates.Enhancer, client: Client):
    files = client.models.text_encoders
    return next((f for f in files if enhancer.file in f.lower()), None)


async def _comfy_generate(
    request: prompt_templates.Request,
    prompt: str,
    extent: Extent | None,
    arch: Arch,
    style: Style,
    client: Client | None,
):
    if client is None:
        raise RuntimeError(_("Not connected to a server."))
    if arch is Arch.auto:
        raise RuntimeError(_("The style's checkpoint was not found on the server."))
    opts = request.options
    seed = random.randint(0, 2**31 - 1)

    # Dedicated prompt enhancer of the model (eg. P-H3-Image's upsampler), if it's on the server
    enhancer = opts.enhancer
    if enhancer and request.source != "custom":
        if file := _find_enhancer(enhancer, client):
            user = prompt_templates.fill(enhancer.user_format, prompt, extent)
            system = request.system if enhancer.system else ""
            raw = enhancer.raw_template
            if raw:
                user = raw.replace("{system}", system).replace("{user}", user)
                system = ""
            log.info(f"Improving prompt with {file} ({enhancer.clip_type})")
            return await client.generate_text(
                file,
                enhancer.clip_type,
                user,
                system,
                seed,
                max_length=opts.max_tokens,
                thinking=enhancer.thinking,
                temperature=opts.temperature,
                sampling=enhancer.sampling,
                use_default_template=raw is None,
            )

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
    log.info(f"Improving prompt with {text_encoder} ({arch.clip_loader_type})")
    return await client.generate_text(
        text_encoder,
        arch.clip_loader_type,
        request.user,
        request.system,
        seed,
        max_length=opts.max_tokens,
        thinking=opts.thinking,
        temperature=opts.temperature,
        sampling=opts.sampling,
    )


_requests: RequestManager | None = None


async def _openai_chat(request: prompt_templates.Request):
    global _requests
    if _requests is None:
        _requests = RequestManager()
    base = settings.prompt_enhancer_url.strip().rstrip("/")
    if not base:
        raise RuntimeError(_("Set the LLM Server URL in Settings > Interface."))
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    messages = [{"role": "user", "content": request.user}]
    if request.system:
        messages.insert(0, {"role": "system", "content": request.system})
    opts = request.options
    data = {
        "model": settings.prompt_enhancer_model.strip() or "default",
        "messages": messages,
        "temperature": opts.temperature,
        "max_tokens": max(opts.max_tokens, 4096 if opts.thinking else 1024),
        "stream": False,
    }
    key = settings.prompt_enhancer_api_key.strip() or None
    log.info(f"Improving prompt with {url} (model {data['model']})")
    reply = await _requests.http("POST", url, data, timeout=600, bearer=key)
    try:
        message = reply["choices"][0]["message"]
        return message.get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(_("Unexpected answer from the LLM server") + f": {str(reply)[:200]}")
