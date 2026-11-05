import pytest

from ai_diffusion.backend import workflow
from ai_diffusion.backend.api import CheckpointInput
from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.comfy_workflow import ComfyWorkflow
from ai_diffusion.backend.resources import ResourceKind, resource_id
from ai_diffusion.files import FileFormat
from ai_diffusion.model import prompt_enhance, prompt_templates
from ai_diffusion.settings import settings
from ai_diffusion.style import Arch, StyleSettings

single_encoder_archs = [
    a for a in Arch.list() if a.supports_text_encoder_override and len(a.text_encoders) == 1
]


def models_for(arch: Arch):
    models = ClientModels()
    models.checkpoints = {
        "model.safetensors": CheckpointInfo("model.safetensors", arch, FileFormat.diffusion)
    }
    for te in ["clip_l", "clip_g", "t5"] + [a.main_text_encoder for a in single_encoder_archs]:
        models.resources[resource_id(ResourceKind.text_encoder, Arch.all, te)] = f"{te}.safetensors"
    models.resources[resource_id(ResourceKind.vae, arch, "default")] = "vae.safetensors"
    return models


def text_encoder_nodes(arch: Arch, text_encoder: str):
    w = ComfyWorkflow()
    checkpoint = CheckpointInput("model.safetensors", arch, text_encoder=text_encoder)
    workflow.load_checkpoint_with_lora(w, checkpoint, models_for(arch))
    loaders = ("CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader")
    return [n["inputs"] for n in w.root.values() if n["class_type"] in loaders]


@pytest.mark.parametrize("arch", single_encoder_archs + [Arch.flux, Arch.chroma])
def test_text_encoder_default(arch: Arch):
    default = StyleSettings.text_encoder.default
    nodes = text_encoder_nodes(arch, default)
    assert len(nodes) == 1
    files = [v for k, v in nodes[0].items() if k.startswith("clip_name")]
    assert f"{arch.main_text_encoder}.safetensors" in files


@pytest.mark.parametrize("arch", single_encoder_archs + [Arch.flux, Arch.chroma])
def test_text_encoder_override(arch: Arch):
    nodes = text_encoder_nodes(arch, "my-abliterated-te.safetensors")
    files = [v for k, v in nodes[0].items() if k.startswith("clip_name")]
    assert "my-abliterated-te.safetensors" in files
    assert f"{arch.main_text_encoder}.safetensors" not in files
    if arch.clip_loader_type and len(arch.text_encoders) == 1:
        assert nodes[0]["type"] == arch.clip_loader_type
    if arch is Arch.flux:
        assert nodes[0]["clip_name1"] == "clip_l.safetensors"  # only T5 is replaced


@pytest.mark.parametrize("arch", [Arch.sd15, Arch.sdxl, Arch.illu])
def test_text_encoder_override_ignored(arch: Arch):
    assert not arch.supports_text_encoder_override
    nodes = text_encoder_nodes(arch, "my-te.safetensors")
    assert all("my-te.safetensors" not in n.values() for n in nodes)


def test_clean_output():
    clean = prompt_enhance.clean_output
    assert clean("<think>hmm</think>\nA red fox.") == "A red fox."
    assert clean('"A red fox in snow"') == "A red fox in snow"
    assert clean("Prompt: a red fox") == "a red fox"
    assert clean("```\na red fox\n```") == "a red fox"
    assert clean("Improved prompt:  a red fox") == "a red fox"


def test_system_prompt():
    settings.prompt_enhancer_instructions = ""
    assert "Danbooru" in prompt_enhance.system_prompt(Arch.illu)
    assert "separated by commas" in prompt_enhance.system_prompt(Arch.sdxl)
    assert "editing" in prompt_enhance.system_prompt(Arch.qwen_e)
    assert "natural language" in prompt_enhance.system_prompt(Arch.moss)
    assert "Output ONLY the prompt" in prompt_enhance.system_prompt(Arch.moss)
    settings.prompt_enhancer_instructions = "keep it short"
    assert prompt_enhance.system_prompt(Arch.moss).endswith("keep it short")
    settings.prompt_enhancer_instructions = ""


@pytest.mark.parametrize("arch", Arch.list())
def test_template_per_arch(arch: Arch):
    generate = prompt_templates.builtin(arch, editing=False)
    edit = prompt_templates.builtin(arch, editing=True)
    assert generate and edit
    if arch.is_edit:
        assert generate == edit  # edit-only: always the edit template
    elif arch.supports_edit:
        assert generate != edit and "editing" in edit
    else:
        assert generate == edit  # no edit mode, always a text-to-image prompt
    if placeholder := workflow._layer_replace.get(arch):
        assert placeholder.format(2) in edit


def test_template_names_model():
    assert "Illustrious" in prompt_templates.builtin(Arch.illu_v, False)
    assert "Flux 2 Klein" in prompt_templates.builtin(Arch.flux2_9b, False)
    assert "Kontext" in prompt_templates.builtin(Arch.flux_k, False)
    assert "Picture 2" in prompt_templates.builtin(Arch.qwen_e_p, False)
    assert "Layered" in prompt_templates.builtin(Arch.qwen_l, False)


def test_template_style_prompt():
    style = "{prompt}, masterpiece, best quality, absurdres"
    prompt = prompt_enhance.system_prompt(Arch.illu, style_prompt=style)
    assert "do not repeat them: masterpiece, best quality, absurdres" in prompt
    assert "do not repeat" not in prompt_enhance.system_prompt(Arch.illu, style_prompt="{prompt}")  # noqa: RUF027


def test_template_user_file(tmp_path):
    (tmp_path / "illu.txt").write_text("my booru template\n", encoding="utf-8")
    (tmp_path / "qwen2-edit.txt").write_text("my qwen2 edit template", encoding="utf-8")
    (tmp_path / "moss.txt").write_text("  ", encoding="utf-8")  # empty: ignored
    assert prompt_templates.get(Arch.illu, False, tmp_path) == "my booru template"
    assert prompt_templates.get(Arch.qwen2, True, tmp_path) == "my qwen2 edit template"
    assert prompt_templates.get(Arch.qwen2, False, tmp_path) == prompt_templates.builtin(
        Arch.qwen2, False
    )
    assert prompt_templates.get(Arch.moss, False, tmp_path) == prompt_templates.builtin(
        Arch.moss, False
    )
    assert prompt_templates.user_file(Arch.qwen_e, False, tmp_path).name == "qwen_e-edit.txt"
