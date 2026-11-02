import pytest

from ai_diffusion.backend import workflow
from ai_diffusion.backend.api import CheckpointInput
from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.comfy_workflow import ComfyWorkflow
from ai_diffusion.backend.resources import ResourceKind, resource_id
from ai_diffusion.files import FileFormat
from ai_diffusion.model import prompt_enhance
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
    assert "comma-separated keywords" in prompt_enhance.system_prompt(Arch.sdxl)
    assert "editing" in prompt_enhance.system_prompt(Arch.qwen_e)
    assert "natural" in prompt_enhance.system_prompt(Arch.moss)
    settings.prompt_enhancer_instructions = "keep it short"
    assert prompt_enhance.system_prompt(Arch.moss).endswith("keep it short")
    settings.prompt_enhancer_instructions = ""
