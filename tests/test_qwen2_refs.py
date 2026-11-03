from pathlib import Path

from ai_diffusion.backend import workflow
from ai_diffusion.backend.api import ConditioningInput, ControlInput, WorkflowKind
from ai_diffusion.backend.client import CheckpointInfo, ClientModels
from ai_diffusion.backend.comfy_workflow import ComfyObjectInfo
from ai_diffusion.backend.resources import ControlMode, ResourceKind, UpscalerName, resource_id
from ai_diffusion.files import FileCollection, FileLibrary
from ai_diffusion.image import Extent, Image
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import Arch, Style

checkpoint = "qwen_image_2.1_int8_convrot.safetensors"


def client_models():
    models = ClientModels()
    models.checkpoints = {
        checkpoint: CheckpointInfo(checkpoint, Arch.qwen2, workflow.FileFormat.diffusion)
    }
    info = {"CLIPLoader": {"input": {"required": {"type": [["qwen_image"]]}}}}
    models.node_inputs = ComfyObjectInfo(info)
    te_id = resource_id(ResourceKind.text_encoder, Arch.all, "qwen_3vl_8b")
    models.resources[te_id] = "qwen3vl_8b_int8_convrot.safetensors"
    models.resources[resource_id(ResourceKind.vae, Arch.qwen2, "default")] = "qwen_image_2.1_vae"
    for upscaler in UpscalerName:
        models.resources[resource_id(ResourceKind.upscaler, Arch.all, upscaler)] = upscaler.value
    return models


def create(kind: WorkflowKind, image: Image | Extent, edit: bool):
    models = client_models()
    style = Style(Path("default.json"))
    style.checkpoints = [checkpoint]
    files = FileLibrary(FileCollection(), FileCollection())
    cond = ConditioningInput("the cat from <image2> on the sofa")
    cond.control = [ControlInput(ControlMode.reference, Image.create(Extent(2560, 1536)))]
    cond.edit_reference = edit
    job = workflow.prepare(kind, image, cond, style, 7, models, files, PerformanceSettings())
    w = workflow.create(job, models)
    return {id: node for id, node in w.root.items()}


def find(nodes: dict, class_type: str):
    return [n for n in nodes.values() if n["class_type"] == class_type]


def source(nodes: dict, link: list):
    return nodes[link[0]]["class_type"]


def test_reference_layers_scaled_to_one_megapixel():
    nodes = create(WorkflowKind.generate, Extent(1024, 1024), edit=False)
    encode = find(nodes, "TextEncodeQwenImage21")[0]["inputs"]
    assert encode["resolution"] == 0
    assert source(nodes, encode["images.image_1"]) == "ImageScaleToTotalPixels"
    scale = find(nodes, "ImageScaleToTotalPixels")[0]["inputs"]
    assert scale["megapixels"] == 1 and scale["resolution_steps"] == 32


def test_edited_image_keeps_its_size():
    nodes = create(WorkflowKind.refine, Image.create(Extent(1024, 1024)), edit=True)
    encode = find(nodes, "TextEncodeQwenImage21")[0]["inputs"]
    assert source(nodes, encode["images.image_1"]) != "ImageScaleToTotalPixels"
    assert source(nodes, encode["images.image_2"]) == "ImageScaleToTotalPixels"
    assert len(find(nodes, "ImageScaleToTotalPixels")) == 1
