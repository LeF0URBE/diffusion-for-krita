"""System prompts for Improve Prompt, one per diffusion architecture.

Each architecture has a template for text-to-image prompts and, if it can edit images, one
for edit instructions. Users can replace any of them with a text file in the user data folder:
`prompt_templates/<arch>.txt` or `prompt_templates/<arch>-edit.txt` (eg. `illu.txt`,
`qwen_e_p-edit.txt`), where <arch> is the internal name used in style files.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from ..backend.resources import Arch

rules = """Rules:
- Keep every subject, detail and style the user asked for. Do not contradict the input.
- Write in English, even if the input is in another language. Text that should appear in the \
image stays in its original language.
- If the input is already detailed, only polish it; do not invent a different scene.
- Output ONLY the prompt. No preface, no explanation, no quotes around it, no Markdown."""


class Template(NamedTuple):
    generate: str
    edit: str | None = None


# Shared building blocks

_quoted_text = (
    "If the image should contain text (signs, titles, labels, speech), write the exact words "
    'in double quotes and say where they appear and in which lettering, eg. a neon sign reading "OPEN".'
)
_positive_only = (
    "The model has no negative prompt: describe what should be there, never write 'no X' or "
    "'without X'."
)
_edit_core = (
    "The user describes a change to an existing image. Rewrite it as a clear, specific "
    "instruction: what to change, where in the image, and what it should look like afterwards. "
    "You cannot see the image: refer to things only with what the user said (eg. 'the cat', "
    "'the person on the left', 'the sky') and never invent details like clothing, colors or "
    "names. Avoid vague pronouns like 'it' when it is unclear what they mean. Then say what "
    "must stay the same, eg. 'Keep the faces, poses and the rest of the image unchanged.' Only "
    "keep things that the change does not affect. Keep it to one to three sentences."
)


def _references(placeholder: str):
    return (
        f"Images are referred to as {placeholder.format(1)} (the canvas), "
        f"{placeholder.format(2)}, {placeholder.format(3)} and so on (reference images). "
        "Keep every such reference exactly as written, and use them to say which element comes "
        "from which image."
    )


def _natural(model: str, length: str, extra: str = ""):
    return (
        f"You improve text prompts for {model}, which understands natural language. Rewrite the "
        f"user's prompt into {length}: start with the main subject, then its appearance, "
        "clothing, pose and expression, the setting, the composition and camera angle, the "
        "lighting, the color palette and the medium or photographic style. Use concrete visual "
        "words; avoid empty praise like 'masterpiece', 'stunning' or '8k'. " + extra
    ).strip()


def _edit(model: str, extra: str = ""):
    return f"You improve instructions for {model}, an AI image editing model. {_edit_core} {extra}"


# Per-architecture templates

_templates: dict[Arch, Template] = {
    Arch.sd15: Template(
        "You improve text prompts for Stable Diffusion 1.5, which reads short, comma-separated "
        "keywords and only the first 75 tokens well. Rewrite the user's prompt as one line of "
        "concise keywords and short phrases, most important first: subject, appearance, pose, "
        "setting, composition, lighting, colors, then art style or photo terms (eg. 'oil "
        "painting', '35mm photo, shallow depth of field'). Keep it under 40 words. Weights "
        "like (red hair:1.2) are allowed for the one or two most important details."
    ),
    Arch.sdxl: Template(
        "You improve text prompts for Stable Diffusion XL. It works best with short descriptive "
        "phrases separated by commas: a one-sentence description of the subject and scene "
        "first, then keywords for details, composition, lighting, colors and the medium or "
        "camera (eg. 'cinematic photo, 85mm, golden hour, film grain' or 'watercolor "
        "illustration, soft edges'). Keep it under 60 words; text inside the image rarely works, "
        "so only keep it if the user asks for it."
    ),
    Arch.illu: Template(
        "You improve text prompts for an Illustrious anime model trained on Danbooru tags. "
        "Rewrite the user's prompt as comma-separated Danbooru tags, lowercase, with spaces "
        "instead of underscores, in this order: subject count (1girl, 2boys, no humans), "
        "character and series names if given (eg. 'hatsune miku, vocaloid'), artist tags if "
        "given, then hair, eyes, body, clothing, accessories, pose, expression, action, "
        "setting, composition (cowboy shot, from above, close-up) and lighting. Use real "
        "Danbooru tags only, no sentences. Keep it under 40 tags."
    ),
    Arch.sd3: Template(
        _natural(
            "Stable Diffusion 3.5",
            "two to four detailed sentences (at most 150 words)",
            _quoted_text,
        )
    ),
    Arch.flux: Template(
        _natural(
            "Flux.1",
            "one flowing paragraph of 40 to 100 words, written like a photo caption",
            _positive_only + " " + _quoted_text,
        )
    ),
    Arch.flux_k: Template(
        "",
        _edit(
            "Flux Kontext",
            "Prefer precise verbs like 'change', 'replace', 'add', 'remove', 'turn into'. For "
            'text use: Replace "old text" with "new text". For a style change, name the style '
            "and its traits (eg. 'Convert to a watercolor painting with soft washes and paper "
            "texture') and say that the composition stays the same.",
        ),
    ),
    Arch.flux2_4b: Template(
        _natural(
            "Flux 2 Klein",
            "one flowing paragraph of 40 to 100 words, ordered subject, action, style, context",
            "Exact colors may be given as hex codes (eg. #1E90FF). "
            + _positive_only
            + " "
            + _quoted_text,
        ),
        _edit("Flux 2 Klein", _references("image {}")),
    ),
    Arch.chroma: Template(
        _natural(
            "Chroma, a Flux-based model that also knows art styles and anime",
            "one detailed paragraph of 50 to 120 words",
            "Name the medium precisely (photo, anime illustration, 3D render, oil painting). "
            + _quoted_text,
        )
    ),
    Arch.qwen: Template(
        _natural(
            "Qwen-Image, which follows long prompts closely and renders text very well",
            "a detailed paragraph of 60 to 150 words; mention the position of each element "
            "(left, right, foreground, background)",
            _quoted_text,
        )
    ),
    Arch.qwen_e: Template(
        "",
        _edit(
            "Qwen-Image-Edit",
            'For text use: Replace "old text" with "new text", and keep the font and color. '
            "It edits the whole image, so say precisely which part changes.",
        ),
    ),
    Arch.qwen_e_p: Template(
        "",
        _edit(
            "Qwen-Image-Edit 2509/2511, which can combine several images",
            _references("Picture {}")
            + ' For text use: Replace "old text" with "new text", and keep the font and color.',
        ),
    ),
    Arch.qwen_l: Template(
        "",
        "You improve captions for Qwen-Image-Layered, which splits an existing image into "
        "separate layers. Rewrite the user's text into a short caption of the whole image (two "
        "to four sentences) that names each distinct element that should become its own layer: "
        "the background, each subject or object, and any text (in double quotes). Describe "
        "what is there, not what to change.",
    ),
    Arch.qwen2: Template(
        _natural(
            "Qwen-Image 2, which follows long prompts closely and renders text very well",
            "a detailed paragraph of 60 to 150 words; mention the position of each element",
            _quoted_text,
        ),
        _edit(
            "Qwen-Image 2",
            _references("<image{}>")
            + ' For text use: Replace "old text" with "new text", and keep the font and color.',
        ),
    ),
    Arch.anima: Template(
        "You improve text prompts for Anima, an anime model that reads Danbooru tags and natural "
        "language together. Rewrite the user's prompt as: comma-separated Danbooru tags "
        "(lowercase, spaces instead of underscores: subject count like 1girl, character and "
        "series names if given, appearance, clothing, pose, expression, setting), then one or "
        "two plain sentences describing the scene, the composition and the lighting. Keep it "
        "under 80 words."
    ),
    Arch.zimage: Template(
        _natural(
            "Z-Image, a fast photorealistic model with a Qwen3 text encoder",
            "one detailed paragraph of 80 to 180 words; long, specific descriptions work best",
            "For photos, name camera, lens and lighting (eg. 'shot on a 50mm lens, soft window "
            "light'). " + _positive_only + " " + _quoted_text,
        )
    ),
    Arch.ernie: Template(
        _natural(
            "ERNIE-Image, which handles detailed scenes, posters and text well",
            "a detailed paragraph of 60 to 150 words",
            _quoted_text,
        )
    ),
    Arch.krea2: Template(
        _natural(
            "Krea 2, a model with a strong photographic and artistic look",
            "one paragraph of 50 to 120 words",
            "Describe the look like a photographer or art director would: light quality, "
            "lens and framing, film stock or medium, mood. Avoid generic AI-art phrases.",
        )
    ),
    Arch.moss: Template(
        _natural(
            "MOSS-Image, which follows long, structured descriptions",
            "a detailed paragraph of 60 to 150 words; give the position of each subject "
            "(left, center, right, foreground, background)",
            _quoted_text,
        )
    ),
    Arch.fibo2: Template(
        _natural(
            "Fibo 2, which was trained on very detailed captions",
            "a detailed, factual paragraph of 80 to 180 words: list every object with its "
            "position, size, color and material, then the background, lighting, camera angle, "
            "depth of field and the medium or style",
            _quoted_text,
        ),
        _edit("Fibo 2", _references("<image_{}>")),
    ),
    Arch.h3_e: Template(
        "",
        _edit("MiniMax-H3", _references("<Picture {}>")),
    ),
    Arch.ph3: Template(
        _natural(
            "P-H3-Image, which follows long, specific prompts",
            "a detailed paragraph of 80 to 200 words",
            _quoted_text,
        ),
        _edit("P-H3-Image", _references("<Picture {}>")),
    ),
}
_templates[Arch.illu_v] = _templates[Arch.illu]
_templates[Arch.flux2_9b] = _templates[Arch.flux2_4b]
_templates[Arch.fibo2_e] = Template("", _templates[Arch.fibo2].edit)

_fallback = Template(
    _natural("an AI image generation model", "one detailed, vivid paragraph"),
    _edit("an AI image editing model"),
)


def is_edit(arch: Arch, editing: bool):
    """Edit-only models always get an instruction, text-to-image-only models never do."""
    return arch.is_edit or (editing and arch.supports_edit)


def builtin(arch: Arch, editing: bool):
    template = _templates.get(arch, _fallback)
    editing = is_edit(arch, editing)
    text = template.edit if editing else template.generate
    return text or (_fallback.edit if editing else _fallback.generate) or ""


def user_file(arch: Arch, editing: bool, folder: Path):
    suffix = "-edit" if is_edit(arch, editing) else ""
    return folder / f"{arch.name}{suffix}.txt"


def get(arch: Arch, editing: bool, folder: Path | None = None):
    """The system prompt for `arch`, from a user file if there is one, without rules."""
    if folder is not None:
        file = user_file(arch, editing, folder)
        try:
            if file.exists() and (text := file.read_text(encoding="utf-8").strip()):
                return text
        except OSError:
            pass
    return builtin(arch, editing)
