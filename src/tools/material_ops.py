"""Material creation, assignment, and property manipulation tools for 3ds Max.

Covers the full material workflow: creating materials by class, assigning them
to objects, setting properties, creating texture maps, writing OSL shaders,
and managing Multi/Sub-Object sub-material slots.
Works with all material/map types: OpenPBR, Arnold (ai_standard_surface),
V-Ray (VRayMtl), Physical, Standard, OSLMap, Bitmaptexture, ai_bump2d,
and any MAXScript-creatable class.
"""

import json
from pathlib import Path
from typing import Optional
from ..server import mcp, client
from ..coerce import StrList
from src.helpers.maxscript import safe_string, safe_value
from .material_detection import (
    _COLOR_CHANNELS,
    _DEFAULT_CHANNEL_PATTERNS,
    _IMAGE_EXTENSIONS,
    _group_texture_files_for_pbr,
    _match_textures_to_channels,
    _renderer_from_material_class,
    _scan_texture_folder,
)
from .material_shell import (
    UBER_OUT_B as _UBER_OUT_B,
    UBER_OUT_COL as _UBER_OUT_COL,
    UBER_OUT_G as _UBER_OUT_G,
    UBER_OUT_R as _UBER_OUT_R,
    build_shell_maxscript,
)


# ---------------------------------------------------------------------------
# Renderer configs and material-builder helpers
# ---------------------------------------------------------------------------

# Renderer wiring configs and slot mappings
_RENDERER_CONFIGS: dict[str, dict] = {
    "arnold": {
        "material_class": "ai_standard_surface",
        "slots": {
            "diffuse":       "base_color_shader",
            "roughness":     "specular_roughness_shader",
            "glossiness":    "specular_roughness_shader",   # + invert
            "metallic":      "metalness_shader",
            "opacity":       "opacity_shader",
            "emission":      "emission_color_shader",
            "translucency":  "transmission_shader",
            "specular":      "specular_color_shader",
        },
        # Normal/bump/displacement handled specially
    },
    "physical": {
        "material_class": "PhysicalMaterial",
        "slots": {
            "diffuse":       "base_color_map",
            "roughness":     "roughness_map",
            "glossiness":    "roughness_map",  # + invert
            "metallic":      "metalness_map",
            "opacity":       "cutout_map",
            "emission":      "emit_color_map",
            "translucency":  "trans_color_map",
            "specular":      "refl_color_map",
        },
    },
    "openpbr": {
        "material_class": "OpenPBRMaterial",
        "slots": {
            "diffuse":      ["base_color_map", "baseColor_map", "basecolor_map", "base_map", "diffuse_map"],
            "roughness":    ["roughness_map", "specular_roughness_map", "base_roughness_map"],
            "glossiness":   ["roughness_map", "specular_roughness_map", "base_roughness_map"],
            "metallic":     ["metalness_map", "metallic_map", "base_metalness_map"],
            "opacity":      ["opacity_map", "cutout_map", "transparency_map"],
            "emission":     ["emission_color_map", "emit_color_map", "emission_map"],
            "translucency": ["transmission_color_map", "trans_color_map", "transmission_map"],
            "specular":     ["specular_color_map", "refl_color_map"],
            "displacement": ["displacement_map"],
        },
    },
    "materialx": {
        "material_class": "OpenPBRMaterial",
        "slots": {
            "diffuse":      ["base_color_map", "baseColor_map", "basecolor_map", "base_map", "diffuse_map"],
            "roughness":    ["roughness_map", "specular_roughness_map", "base_roughness_map"],
            "glossiness":   ["roughness_map", "specular_roughness_map", "base_roughness_map"],
            "metallic":     ["metalness_map", "metallic_map", "base_metalness_map"],
            "opacity":      ["opacity_map", "cutout_map", "transparency_map"],
            "emission":     ["emission_color_map", "emit_color_map", "emission_map"],
            "translucency": ["transmission_color_map", "trans_color_map", "transmission_map"],
            "specular":     ["specular_color_map", "refl_color_map"],
            "displacement": ["displacement_map"],
        },
    },
    "redshift": {
        "material_class": "RS_Standard_Material",
        "slots": {
            "diffuse":       "base_color_map",
            "roughness":     "refl_roughness_map",
            "glossiness":    "refl_roughness_map",  # + invert
            "metallic":      "metalness_map",
            "opacity":       "opacity_color_map",
            "emission":      "emission_color_map",
            "translucency":  "refr_color_map",
            "specular":      "refl_color_map",
        },
    },
    "vray": {
        "material_class": "VRayMtl",
        "slots": {
            "diffuse":       "texmap_diffuse",
            "roughness":     "texmap_roughness",
            "glossiness":    "texmap_reflectionGlossiness",
            "metallic":      "texmap_metalness",
            "opacity":       "texmap_opacity",
            "emission":      "texmap_self_illumination",
            "translucency":  "texmap_translucent",
            "specular":      "texmap_reflection",
        },
    },
    "octane_standard": {
        "material_class": "Std_Surface_Mtl",
        "slots": {
            "diffuse":       "baseColor_tex",
            "roughness":     "roughness_tex",
            "glossiness":    "roughness_tex",
            "metallic":      "metallic_tex",
            "opacity":       "opacity_tex",
            "emission":      "emissionColor_tex",
            "translucency":  "transmissionColor_tex",
            "specular":      "specularColor_tex",
        },
    },
    "octane_pbr": {
        "material_class": "Open_PBR_Surf__Mtl",
        "slots": {
            "diffuse":       "baseColor_tex",
            "roughness":     "roughness_tex",
            "glossiness":    "roughness_tex",
            "metallic":      "metallic_tex",
            "opacity":       "opacity_tex",
            "emission":      "emissionColor_tex",
            "translucency":  "transmissionColor_tex",
            "specular":      "specularColor_tex",
        },
    },
    "octane_universal": {
        "material_class": "Universal_material",
        "slots": {
            "diffuse":       "albedo_tex",
            "roughness":     "roughness_tex",
            "glossiness":    "roughness_tex",
            "metallic":      "metallic_tex",
            "opacity":       "opacity_tex",
            "emission":      "emission",
            "translucency":  "transmission_tex",
            "specular":      "specular_tex",
        },
    },
}

_PBR_SLOT_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "openpbr": {
        "diffuse":      ["base_color_map", "baseColor_map", "basecolor_map", "base_map", "diffuse_map"],
        "ao":           ["base_color_map", "baseColor_map", "basecolor_map", "base_map", "diffuse_map"],
        "roughness":    ["roughness_map", "specular_roughness_map", "base_roughness_map"],
        "glossiness":   ["roughness_map", "specular_roughness_map", "base_roughness_map"],
        "metallic":     ["metalness_map", "metallic_map", "base_metalness_map"],
        "normal":       ["bump_map", "normal_map"],
        "bump":         ["bump_map", "normal_map"],
        "displacement": ["displacement_map"],
        "opacity":      ["opacity_map", "cutout_map", "transparency_map"],
        "emission":     ["emission_color_map", "emit_color_map", "emission_map"],
        "translucency": ["transmission_color_map", "trans_color_map", "transmission_map"],
        "specular":     ["specular_color_map", "refl_color_map"],
    },
    "physical": {
        "diffuse":      ["base_color_map"],
        "ao":           ["base_color_map"],
        "roughness":    ["roughness_map"],
        "glossiness":   ["roughness_map"],
        "metallic":     ["metalness_map"],
        "normal":       ["bump_map"],
        "bump":         ["bump_map"],
        "displacement": ["displacement_map"],
        "opacity":      ["cutout_map"],
        "emission":     ["emit_color_map", "emission_map"],
        "translucency": ["trans_color_map", "transparency_map"],
        "specular":     ["refl_color_map"],
    },
    "materialx": {
        "diffuse":      ["base_color_map", "baseColor_map", "basecolor_map", "base_map", "diffuse_map"],
        "ao":           ["base_color_map", "baseColor_map", "basecolor_map", "base_map", "diffuse_map"],
        "roughness":    ["roughness_map", "specular_roughness_map", "base_roughness_map"],
        "glossiness":   ["roughness_map", "specular_roughness_map", "base_roughness_map"],
        "metallic":     ["metalness_map", "metallic_map", "base_metalness_map"],
        "normal":       ["bump_map", "normal_map"],
        "bump":         ["bump_map", "normal_map"],
        "displacement": ["displacement_map"],
        "opacity":      ["opacity_map", "cutout_map", "transparency_map"],
        "emission":     ["emission_color_map", "emit_color_map", "emission_map"],
        "translucency": ["transmission_color_map", "trans_color_map", "transmission_map"],
        "specular":     ["specular_color_map", "refl_color_map"],
    },
    "arnold": {
        "diffuse":      ["base_color_shader"],
        "ao":           ["base_color_shader"],
        "roughness":    ["specular_roughness_shader"],
        "glossiness":   ["specular_roughness_shader"],
        "metallic":     ["metalness_shader"],
        "normal":       ["normal_shader"],
        "bump":         ["normal_shader"],
        "opacity":      ["opacity_shader"],
        "emission":     ["emission_color_shader"],
        "translucency": ["transmission_shader"],
        "specular":     ["specular_color_shader"],
    },
    "redshift": {
        "diffuse":      ["base_color_map"],
        "ao":           ["base_color_map"],
        "roughness":    ["refl_roughness_map"],
        "glossiness":   ["refl_roughness_map"],
        "metallic":     ["metalness_map"],
        "normal":       ["bump_input"],
        "bump":         ["bump_input"],
        "displacement": ["displacement_input"],
        "opacity":      ["opacity_color_map"],
        "emission":     ["emission_color_map"],
        "translucency": ["refr_color_map"],
        "specular":     ["refl_color_map"],
    },
    "vray": {
        "diffuse":      ["texmap_diffuse", "diffuse_texmap", "diffuseMap"],
        "ao":           ["texmap_diffuse", "diffuse_texmap", "diffuseMap"],
        "roughness":    ["texmap_roughness", "roughness_texmap", "texmap_reflectionRoughness", "reflectionRoughness_texmap", "reflection_roughness_texmap"],
        "glossiness":   ["texmap_reflectionGlossiness", "reflectionGlossiness_texmap", "reflection_glossiness_texmap"],
        "metallic":     ["texmap_metalness", "metalness_texmap", "texmap_metallic", "metallic_texmap"],
        "normal":       ["texmap_bump", "bump_texmap", "bumpMap"],
        "bump":         ["texmap_bump", "bump_texmap", "bumpMap"],
        "displacement": ["texmap_displacement", "displacement_texmap", "displacementMap"],
        "opacity":      ["texmap_opacity", "opacity_texmap", "opacityMap"],
        "emission":     ["texmap_self_illumination", "selfIllumination_texmap", "self_illumination_texmap"],
        "translucency": ["texmap_translucent", "translucent_texmap", "texmap_translucency"],
        "specular":     ["texmap_reflection", "reflection_texmap", "reflectionMap"],
    },
    "octane_standard": {
        "diffuse":      ["baseColor_tex", "albedo_tex"],
        "ao":           ["baseColor_tex", "albedo_tex"],
        "roughness":    ["roughness_tex"],
        "glossiness":   ["roughness_tex"],
        "metallic":     ["metallic_tex"],
        "normal":       ["normal_tex"],
        "bump":         ["bump_tex"],
        "displacement": ["displacement"],
        "opacity":      ["opacity_tex"],
        "emission":     ["emissionColor_tex", "emission"],
        "translucency": ["transmissionColor_tex", "transmission_tex"],
        "specular":     ["specularColor_tex", "specular_tex"],
    },
    "octane_pbr": {
        "diffuse":      ["baseColor_tex"],
        "ao":           ["baseColor_tex"],
        "roughness":    ["roughness_tex"],
        "glossiness":   ["roughness_tex"],
        "metallic":     ["metallic_tex"],
        "normal":       ["normal_tex"],
        "bump":         ["bump_tex"],
        "displacement": ["displacement"],
        "opacity":      ["opacity_tex"],
        "emission":     ["emissionColor_tex", "emission"],
        "translucency": ["transmissionColor_tex"],
        "specular":     ["specularColor_tex", "specular_tex"],
    },
    "octane_universal": {
        "diffuse":      ["albedo_tex"],
        "ao":           ["albedo_tex"],
        "roughness":    ["roughness_tex"],
        "glossiness":   ["roughness_tex"],
        "metallic":     ["metallic_tex"],
        "normal":       ["normal_tex"],
        "bump":         ["bump_tex"],
        "displacement": ["displacement"],
        "opacity":      ["opacity_tex"],
        "emission":     ["emission"],
        "translucency": ["transmission_tex"],
        "specular":     ["specular_tex"],
    },
}

# Octane Channel_picker.colorChannel values for ORM split (R/G/B)
_OCTANE_CH_R = 0
_OCTANE_CH_G = 1
_OCTANE_CH_B = 2


def _ms_path(p: Path) -> str:
    """Convert a Path to a MAXScript-safe forward-slash string."""
    return str(p).replace("\\", "/")


def _ms_name_array(values: list[str]) -> str:
    return "#(" + ", ".join(f'"{safe_string(v)}"' for v in values) + ")"


def _material_slot_hints(material_class: str) -> dict[str, str]:
    """Return compact map-class hints by material class."""
    cls = material_class.lower()
    if cls == "ai_standard_surface":
        return {
            "preferredBitmapClass": "ai_image",
            "normalHelperClass": "ai_normal_map",
            "bumpHelperClass": "ai_bump2d",
        }
    if cls == "rs_standard_material":
        return {
            "preferredBitmapClass": "Bitmaptexture",
            "normalHelperClass": "RS_BumpMap",
            "bumpHelperClass": "RS_BumpMap",
        }
    if cls in {"vraymtl", "v_ray_mtl", "vray_mtl"}:
        return {
            "preferredBitmapClass": "Bitmaptexture",
            "normalHelperClass": "Normal_Bump",
            "bumpHelperClass": "Normal_Bump",
        }
    if cls in {"openpbrmaterial", "openpbr_material", "physicalmaterial", "standardmaterial", "gltfmaterial", "maxusdpreviewsurface"}:
        return {
            "preferredBitmapClass": "Bitmaptexture",
            "normalHelperClass": "Normal_Bump",
            "bumpHelperClass": "Normal_Bump",
        }
    if cls in {"std_surface_mtl", "open_pbr_surf__mtl", "open_pbr_surf_mtl",
               "universal_material"} or "std_surface_mtl" in cls or cls.startswith("octane"):
        return {
            "preferredBitmapClass": "Image_MTX",
            "normalHelperClass": "",
            "bumpHelperClass": "",
            "channelPickerClass": "Channel_picker",
            "compositeMultiplyClass": "Multiply_MTX",
            "invertClass": "Invert_MTX",
            "texInputTypeFlag": "_input_type=2",
        }
    return {
        "preferredBitmapClass": "Bitmaptexture",
        "normalHelperClass": "",
        "bumpHelperClass": "",
    }


def _truncate_slots(payload: dict, key: str, max_per_group: int, out: dict, trunc: dict) -> None:
    items = payload.get(key, [])
    if not isinstance(items, list):
        out[key] = []
        return
    out[key] = items[:max_per_group]
    if len(items) > max_per_group:
        trunc[key] = len(items)


def _build_arnold_maxscript(
    matched: dict[str, Path],
    material_name: str,
    assign_to: list[str] | None,
) -> str:
    """Generate MAXScript for Arnold (ai_standard_surface) material setup."""
    lines: list[str] = []
    safe_mat = safe_string(material_name)
    lines.append(f'mat = ai_standard_surface name:"{safe_mat}"')
    lines.append('summary = "Arnold ai_standard_surface"')
    lines.append('channelList = ""')

    for channel, fpath in matched.items():
        var = f"bm_{channel}"
        fp = _ms_path(fpath)
        is_color = channel in _COLOR_CHANNELS
        cs = "sRGB" if is_color else "Raw"

        # Create ai_image bitmap
        lines.append(f'{var} = ai_image name:"{channel}" filename:"{fp}" color_space:"{cs}"')

        if channel == "diffuse":
            # Check if AO exists to composite
            if "ao" in matched:
                ao_fp = _ms_path(matched["ao"])
                lines.append(f'bm_ao = ai_image name:"ao" filename:"{ao_fp}" color_space:"Raw"')
                lines.append('comp = ai_layer_rgba name:"Diffuse_AO"')
                lines.append(f'comp.input1_shader = {var}')
                lines.append('comp.enable2 = true')
                lines.append('comp.input2_shader = bm_ao')
                lines.append('comp.operation2 = 5')  # multiply (layer 2)
                lines.append('mat.base_color_shader = comp')
                lines.append('channelList += "diffuse(+ao), "')
            else:
                lines.append(f'mat.base_color_shader = {var}')
                lines.append('channelList += "diffuse, "')
        elif channel == "ao":
            # Handled inside diffuse block above; skip standalone
            continue
        elif channel == "glossiness":
            lines.append(f'inv = ai_color_correct name:"GlossToRough" input_shader:{var}')
            lines.append('inv.invert = true')
            lines.append('mat.specular_roughness_shader = inv')
            lines.append('channelList += "glossiness(inverted), "')
        elif channel == "normal":
            lines.append(f'nrmMap = ai_normal_map name:"NormalMap" input_shader:{var}')
            if "bump" in matched:
                bump_fp = _ms_path(matched["bump"])
                lines.append(f'bm_bump_h = ai_image name:"bump" filename:"{bump_fp}" color_space:"Raw"')
                lines.append('bmpNode = ai_bump2d name:"Bump"')
                lines.append('bmpNode.bump_map_shader = bm_bump_h')
                lines.append('bmpNode.normal_shader = nrmMap')
                lines.append('mat.normal_shader = bmpNode')
                lines.append('channelList += "normal(+bump), "')
            else:
                lines.append('bmpNode = ai_bump2d name:"NormalBump"')
                lines.append('bmpNode.normal_shader = nrmMap')
                lines.append('mat.normal_shader = bmpNode')
                lines.append('channelList += "normal, "')
        elif channel == "bump":
            # Handled inside normal block if normal exists
            if "normal" not in matched:
                lines.append('bmpNode = ai_bump2d name:"Bump"')
                lines.append(f'bmpNode.bump_map_shader = {var}')
                lines.append('mat.normal_shader = bmpNode')
                lines.append('channelList += "bump, "')
        elif channel == "displacement":
            # Displacement is modifier-based, note it but don't wire
            lines.append('channelList += "displacement(skipped-modifier-based), "')
        elif channel == "ior":
            lines.append('channelList += "ior(skipped-no-map-slot), "')
        else:
            # Standard slot wiring
            slot = _RENDERER_CONFIGS["arnold"]["slots"].get(channel)
            if slot:
                lines.append(f'mat.{slot} = {var}')
                lines.append(f'channelList += "{channel}, "')

    # Assign to objects
    if assign_to:
        names_arr = "#(" + ", ".join(f'"{safe_string(n)}"' for n in assign_to) + ")"
        lines.append(f'nameList = {names_arr}')
        lines.append('assignCount = 0')
        lines.append('for n in nameList do (obj = getNodeByName n; if obj != undefined then (obj.material = mat; assignCount += 1))')
        lines.append('summary += " | Assigned to " + (assignCount as string) + " object(s)"')

    lines.append('summary += " | Channels: " + channelList')
    lines.append('summary')

    return "(\n    " + "\n    ".join(lines) + "\n)"


def _build_physical_maxscript(
    matched: dict[str, Path],
    material_name: str,
    assign_to: list[str] | None,
) -> str:
    """Generate MAXScript for PhysicalMaterial setup."""
    lines: list[str] = []
    safe_mat = safe_string(material_name)
    lines.append(f'mat = PhysicalMaterial name:"{safe_mat}"')
    lines.append('summary = "PhysicalMaterial"')
    lines.append('channelList = ""')

    for channel, fpath in matched.items():
        var = f"bm_{channel}"
        fp = _ms_path(fpath)

        # Create Bitmaptexture
        lines.append(f'{var} = Bitmaptexture name:"{channel}" fileName:"{fp}"')

        if channel == "diffuse":
            if "ao" in matched:
                ao_fp = _ms_path(matched["ao"])
                lines.append(f'bm_ao = Bitmaptexture name:"ao" fileName:"{ao_fp}"')
                lines.append('comp = CompositeTexturemap()')
                lines.append('comp.name = "Diffuse_AO"')
                lines.append(f'comp.mapList[1] = {var}')
                lines.append('comp.mapList[2] = bm_ao')
                lines.append('comp.blendMode[2] = 5')  # multiply
                lines.append('mat.base_color_map = comp')
                lines.append('channelList += "diffuse(+ao), "')
            else:
                lines.append('mat.base_color_map = ' + var)
                lines.append('channelList += "diffuse, "')
        elif channel == "ao":
            continue
        elif channel == "glossiness":
            lines.append(f'inv = Output name:"GlossToRough"')
            lines.append(f'inv.map1 = {var}')
            lines.append('inv.output.invert = true')
            lines.append('mat.roughness_map = inv')
            lines.append('channelList += "glossiness(inverted), "')
        elif channel == "normal":
            lines.append(f'nrmBump = Normal_Bump name:"NormalBump"')
            lines.append(f'nrmBump.normal_map = {var}')
            if "bump" in matched:
                bump_fp = _ms_path(matched["bump"])
                lines.append(f'bm_bump_h = Bitmaptexture name:"bump" fileName:"{bump_fp}"')
                lines.append('nrmBump.bump_map = bm_bump_h')
                lines.append('mat.bump_map = nrmBump')
                lines.append('channelList += "normal(+bump), "')
            else:
                lines.append('mat.bump_map = nrmBump')
                lines.append('channelList += "normal, "')
        elif channel == "bump":
            if "normal" not in matched:
                lines.append(f'nrmBump = Normal_Bump name:"BumpOnly"')
                lines.append(f'nrmBump.bump_map = {var}')
                lines.append('mat.bump_map = nrmBump')
                lines.append('channelList += "bump, "')
        elif channel == "displacement":
            lines.append(f'mat.displacement_map = {var}')
            lines.append('channelList += "displacement, "')
        elif channel == "ior":
            lines.append('channelList += "ior(skipped-no-map-slot), "')
        else:
            slot = _RENDERER_CONFIGS["physical"]["slots"].get(channel)
            if slot:
                lines.append(f'mat.{slot} = {var}')
                lines.append(f'channelList += "{channel}, "')

    if assign_to:
        names_arr = "#(" + ", ".join(f'"{safe_string(n)}"' for n in assign_to) + ")"
        lines.append(f'nameList = {names_arr}')
        lines.append('assignCount = 0')
        lines.append('for n in nameList do (obj = getNodeByName n; if obj != undefined then (obj.material = mat; assignCount += 1))')
        lines.append('summary += " | Assigned to " + (assignCount as string) + " object(s)"')

    lines.append('summary += " | Channels: " + channelList')
    lines.append('summary')

    return "(\n    " + "\n    ".join(lines) + "\n)"


def _build_openpbr_maxscript(
    matched: dict[str, Path],
    material_name: str,
    assign_to: list[str] | None,
) -> str:
    """Generate MAXScript for OpenPBR material setup with Physical fallback."""
    lines: list[str] = []
    safe_mat = safe_string(material_name)
    lines.extend([
        "fn mcp_setFirstMap target propNames tex = (",
        "    for propName in propNames do (",
        "        try (setProperty target (propName as name) tex; return propName) catch ()",
        "    )",
        "    undefined",
        ")",
        "fn mcp_createOpenPbrPreferred matName = (",
        "    local m = undefined",
        '    try (m = OpenPBRMaterial name:matName) catch ()',
        '    if m == undefined do try (m = OpenPBR_Material name:matName) catch ()',
        '    if m == undefined do try (m = OpenPBR_Mtl name:matName) catch ()',
        '    if m == undefined do try (m = PhysicalMaterial name:matName) catch ()',
        '    if m == undefined do throw "OpenPBRMaterial/OpenPBR_Material/OpenPBR_Mtl/PhysicalMaterial are unavailable"',
        "    m",
        ")",
        f'mat = mcp_createOpenPbrPreferred "{safe_mat}"',
        'summary = ((classOf mat) as string)',
        'if matchPattern summary pattern:"Physical*" do summary += " (fallback; OpenPBR class unavailable)"',
        'channelList = ""',
        'skippedList = ""',
    ])

    slots = _RENDERER_CONFIGS["openpbr"]["slots"]

    def ms_name_array(values: list[str]) -> str:
        return "#(" + ", ".join(f'"{safe_string(v)}"' for v in values) + ")"

    for channel, fpath in matched.items():
        var = f"bm_{channel}"
        fp = _ms_path(fpath)
        lines.append(f'{var} = Bitmaptexture name:"{channel}" fileName:"{fp}"')

        if channel == "diffuse":
            if "ao" in matched:
                ao_fp = _ms_path(matched["ao"])
                lines.append(f'bm_ao = Bitmaptexture name:"ao" fileName:"{ao_fp}"')
                lines.append('comp = CompositeTexturemap()')
                lines.append('comp.name = "Diffuse_AO"')
                lines.append(f'comp.mapList[1] = {var}')
                lines.append('comp.mapList[2] = bm_ao')
                lines.append('comp.blendMode[2] = 5')
                lines.append(f'slotName = mcp_setFirstMap mat {ms_name_array(slots["diffuse"])} comp')
                lines.append('if slotName != undefined then channelList += "diffuse(+ao)->" + slotName + ", " else skippedList += "diffuse, "')
            else:
                lines.append(f'slotName = mcp_setFirstMap mat {ms_name_array(slots["diffuse"])} {var}')
                lines.append('if slotName != undefined then channelList += "diffuse->" + slotName + ", " else skippedList += "diffuse, "')
        elif channel == "ao":
            continue
        elif channel == "glossiness":
            lines.append('inv = Output name:"GlossToRough"')
            lines.append(f'inv.map1 = {var}')
            lines.append('inv.output.invert = true')
            lines.append(f'slotName = mcp_setFirstMap mat {ms_name_array(slots["glossiness"])} inv')
            lines.append('if slotName != undefined then channelList += "glossiness(inverted)->" + slotName + ", " else skippedList += "glossiness, "')
        elif channel == "normal":
            lines.append('nrmBump = Normal_Bump name:"NormalBump"')
            lines.append(f'nrmBump.normal_map = {var}')
            if "bump" in matched:
                bump_fp = _ms_path(matched["bump"])
                lines.append(f'bm_bump_h = Bitmaptexture name:"bump" fileName:"{bump_fp}"')
                lines.append('nrmBump.bump_map = bm_bump_h')
                lines.append('slotName = mcp_setFirstMap mat #("bump_map", "normal_map") nrmBump')
                lines.append('if slotName != undefined then channelList += "normal(+bump)->" + slotName + ", " else skippedList += "normal, "')
            else:
                lines.append('slotName = mcp_setFirstMap mat #("bump_map", "normal_map") nrmBump')
                lines.append('if slotName != undefined then channelList += "normal->" + slotName + ", " else skippedList += "normal, "')
        elif channel == "bump":
            if "normal" not in matched:
                lines.append('nrmBump = Normal_Bump name:"BumpOnly"')
                lines.append(f'nrmBump.bump_map = {var}')
                lines.append('slotName = mcp_setFirstMap mat #("bump_map", "normal_map") nrmBump')
                lines.append('if slotName != undefined then channelList += "bump->" + slotName + ", " else skippedList += "bump, "')
        elif channel == "ior":
            lines.append('skippedList += "ior, "')
        else:
            candidates = slots.get(channel)
            if candidates:
                lines.append(f'slotName = mcp_setFirstMap mat {ms_name_array(candidates)} {var}')
                lines.append(f'if slotName != undefined then channelList += "{channel}->" + slotName + ", " else skippedList += "{channel}, "')

    if assign_to:
        names_arr = "#(" + ", ".join(f'"{safe_string(n)}"' for n in assign_to) + ")"
        lines.append(f'nameList = {names_arr}')
        lines.append('assignCount = 0')
        lines.append('for n in nameList do (obj = getNodeByName n; if obj != undefined then (obj.material = mat; assignCount += 1))')
        lines.append('summary += " | Assigned to " + (assignCount as string) + " object(s)"')

    lines.append('summary += " | Channels: " + channelList')
    lines.append('if skippedList != "" do summary += " | Skipped: " + skippedList')
    lines.append('summary')

    return "(\n    " + "\n    ".join(lines) + "\n)"


def _build_redshift_maxscript(
    matched: dict[str, Path],
    material_name: str,
    assign_to: list[str] | None,
) -> str:
    """Generate MAXScript for Redshift (RS_Standard_Material) setup."""
    lines: list[str] = []
    safe_mat = safe_string(material_name)
    lines.append(f'mat = RS_Standard_Material name:"{safe_mat}"')
    lines.append('summary = "Redshift RS_Standard_Material"')
    lines.append('channelList = ""')

    for channel, fpath in matched.items():
        var = f"bm_{channel}"
        fp = _ms_path(fpath)

        lines.append(f'{var} = Bitmaptexture name:"{channel}" fileName:"{fp}"')

        if channel == "diffuse":
            if "ao" in matched:
                ao_fp = _ms_path(matched["ao"])
                lines.append(f'bm_ao = Bitmaptexture name:"ao" fileName:"{ao_fp}"')
                lines.append('comp = CompositeTexturemap()')
                lines.append('comp.name = "Diffuse_AO"')
                lines.append(f'comp.mapList[1] = {var}')
                lines.append('comp.mapList[2] = bm_ao')
                lines.append('comp.blendMode[2] = 5')
                lines.append('mat.base_color_map = comp')
                lines.append('channelList += "diffuse(+ao), "')
            else:
                lines.append(f'mat.base_color_map = {var}')
                lines.append('channelList += "diffuse, "')
        elif channel == "ao":
            continue
        elif channel == "glossiness":
            lines.append(f'inv = Output name:"GlossToRough"')
            lines.append(f'inv.map1 = {var}')
            lines.append('inv.output.invert = true')
            lines.append('mat.refl_roughness_map = inv')
            lines.append('channelList += "glossiness(inverted), "')
        elif channel == "normal":
            lines.append('rsBump = RS_BumpMap name:"NormalBump"')
            lines.append(f'rsBump.input_map = {var}')
            lines.append('rsBump.inputType = 1')  # tangent-space normal
            if "bump" in matched:
                bump_fp = _ms_path(matched["bump"])
                lines.append(f'bm_bump_h = Bitmaptexture name:"bump" fileName:"{bump_fp}"')
                # Redshift: chain bump into the bump map input
                lines.append('rsBumpH = RS_BumpMap name:"BumpHeight"')
                lines.append('rsBumpH.input_map = bm_bump_h')
                lines.append('rsBumpH.inputType = 0')  # bump
                lines.append('-- Redshift: wire normal to bump_input, height bump separate')
                lines.append('mat.bump_input = rsBump')
                lines.append('channelList += "normal(+bump partially), "')
            else:
                lines.append('mat.bump_input = rsBump')
                lines.append('channelList += "normal, "')
        elif channel == "bump":
            if "normal" not in matched:
                lines.append('rsBump = RS_BumpMap name:"Bump"')
                lines.append(f'rsBump.input_map = {var}')
                lines.append('rsBump.inputType = 0')
                lines.append('mat.bump_input = rsBump')
                lines.append('channelList += "bump, "')
        elif channel == "displacement":
            lines.append(f'mat.displacement_input = {var}')
            lines.append('channelList += "displacement, "')
        elif channel == "ior":
            lines.append('channelList += "ior(skipped-no-map-slot), "')
        else:
            slot = _RENDERER_CONFIGS["redshift"]["slots"].get(channel)
            if slot:
                lines.append(f'mat.{slot} = {var}')
                lines.append(f'channelList += "{channel}, "')

    if assign_to:
        names_arr = "#(" + ", ".join(f'"{safe_string(n)}"' for n in assign_to) + ")"
        lines.append(f'nameList = {names_arr}')
        lines.append('assignCount = 0')
        lines.append('for n in nameList do (obj = getNodeByName n; if obj != undefined then (obj.material = mat; assignCount += 1))')
        lines.append('summary += " | Assigned to " + (assignCount as string) + " object(s)"')

    lines.append('summary += " | Channels: " + channelList')
    lines.append('summary')

    return "(\n    " + "\n    ".join(lines) + "\n)"


@mcp.tool()
def assign_material(
    names: StrList,
    material_class: str,
    material_name: str = "",
    params: str = "",
) -> str:
    """Create a material and assign it to one or more objects."""
    if client.native_available:
        payload = {
            "names": names,
            "material_class": material_class,
            "material_name": material_name,
            "params": params,
        }
        response = client.send_command(json.dumps(payload), cmd_type="native:assign_material")
        return response.get("result", "")

    safe_mat_name = safe_string(material_name)
    name_param = f' name:"{safe_mat_name}"' if material_name else ""
    name_arr = "#(" + ", ".join(f'"{safe_string(n)}"' for n in names) + ")"

    maxscript = f"""(
        try (
            mat = {material_class}{name_param} {params}
            nameList = {name_arr}
            assignCount = 0
            notFound = #()
            for n in nameList do (
                obj = getNodeByName n
                if obj != undefined then (
                    obj.material = mat
                    assignCount += 1
                ) else (
                    append notFound n
                )
            )
            msg = "Created " + (classof mat) as string + " \\\"" + mat.name + "\\\" and assigned to " + (assignCount as string) + " object(s)"
            if notFound.count > 0 do msg += " | Not found: " + (notFound as string)
            msg
        ) catch (
            "Error: " + (getCurrentException())
        )
    )"""
    response = client.send_command(maxscript)
    return response.get("result", "")


@mcp.tool()
def set_material_property(
    name: str,
    property: str,
    value: str,
    sub_material_index: int = 0,
) -> str:
    """Set a property on an object's material (or sub-material)."""
    if client.native_available:
        payload = {
            "name": name,
            "property": property,
            "value": value,
            "sub_material_index": sub_material_index,
        }
        response = client.send_command(json.dumps(payload), cmd_type="native:set_material_property")
        return response.get("result", "")

    safe = safe_string(name)
    safe_prop = safe_string(property)

    if sub_material_index > 0:
        mat_expr = f"obj.material[{sub_material_index}]"
        mat_label = f"sub-material [{sub_material_index}]"
    else:
        mat_expr = "obj.material"
        mat_label = "material"

    maxscript = f"""(
        obj = getNodeByName "{safe}"
        if obj == undefined then (
            "Object not found: {safe}"
        ) else if obj.material == undefined then (
            "No material assigned to {safe}"
        ) else (
            mat = {mat_expr}
            if mat == undefined then (
                "Sub-material index {sub_material_index} not found on {safe}"
            ) else (
                try (
                    mat.{safe_prop} = {safe_value(value)}
                    readback = (getproperty mat #{safe_prop}) as string
                    "Set " + mat.name + ".{safe_prop} = " + readback
                ) catch (
                    "Error setting {safe_prop}: " + (getCurrentException())
                )
            )
        )
    )"""
    response = client.send_command(maxscript)
    return response.get("result", "")


@mcp.tool()
def set_material_properties(
    name: str,
    properties: dict[str, str],
    sub_material_index: int = 0,
) -> str:
    """Set multiple properties on an object's material in a single call."""
    if client.native_available:
        payload = {
            "name": name,
            "properties": properties,
            "sub_material_index": sub_material_index,
        }
        response = client.send_command(json.dumps(payload), cmd_type="native:set_material_properties")
        return response.get("result", "")

    safe = safe_string(name)

    if sub_material_index > 0:
        mat_expr = f"obj.material[{sub_material_index}]"
    else:
        mat_expr = "obj.material"

    # Build the property-setting lines
    set_lines = []
    for prop, val in properties.items():
        safe_prop = safe_string(prop)
        set_lines.append(
            f'try (mat.{safe_prop} = {safe_value(val)}; append okList "{safe_prop}") '
            f'catch (append errList ("{safe_prop}: " + (getCurrentException())))'
        )
    set_block = "\n            ".join(set_lines)

    maxscript = f"""(
        obj = getNodeByName "{safe}"
        if obj == undefined then (
            "Object not found: {safe}"
        ) else if obj.material == undefined then (
            "No material assigned to {safe}"
        ) else (
            mat = {mat_expr}
            if mat == undefined then (
                "Sub-material index {sub_material_index} not found on {safe}"
            ) else (
                okList = #()
                errList = #()
                {set_block}
                msg = "Set " + (okList.count as string) + " properties on " + mat.name
                if okList.count > 0 do (
                    msg += ": "
                    for i = 1 to okList.count do (
                        if i > 1 do msg += ", "
                        msg += okList[i]
                    )
                )
                if errList.count > 0 do (
                    msg += " | Errors: "
                    for i = 1 to errList.count do (
                        if i > 1 do msg += "; "
                        msg += errList[i]
                    )
                )
                msg
            )
        )
    )"""
    response = client.send_command(maxscript)
    return response.get("result", "")


@mcp.tool()
def get_material_slots(
    name: str,
    sub_material_index: int = 0,
    include_values: bool = False,
    max_slots: int = 40,
    slot_scope: str = "map",
    max_per_group: int = 15,
) -> str:
    """Get compact material slot/property info without schema caches."""
    if client.native_available:
        try:
            payload = json.dumps({
                "name": name,
                "sub_material_index": sub_material_index,
                "include_values": include_values,
                "max_slots": max(1, int(max_slots)),
                "slot_scope": (slot_scope or "map").strip().lower(),
                "max_per_group": max(1, int(max_per_group)),
            })
            response = client.send_command(payload, cmd_type="native:get_material_slots")
            raw = response.get("result", "")
            if not raw:
                return raw
            try:
                payload_data = json.loads(raw)
            except Exception:
                return raw
            if isinstance(payload_data, dict):
                material_class = str(payload_data.get("class", ""))
                payload_data["hints"] = _material_slot_hints(material_class)
            return json.dumps(payload_data, separators=(",", ":"))
        except RuntimeError:
            pass

    safe = safe_string(name)
    max_slots = max(1, int(max_slots))
    max_per_group = max(1, int(max_per_group))
    slot_scope = (slot_scope or "map").strip().lower()
    if slot_scope not in {"map", "summary", "all"}:
        slot_scope = "map"
    include_vals = "true" if include_values else "false"

    if sub_material_index > 0:
        mat_expr = f"obj.material[{sub_material_index}]"
    else:
        mat_expr = "obj.material"

    maxscript = f"""(
        local esc = MCP_Server.escapeJsonString

        fn toJsonNameArray arr = (
            local out = "["
            local q = (bit.intAsChar 34)
            for i = 1 to arr.count do (
                if i > 1 do out += ","
                out += q + (esc arr[i]) + q
            )
            out += "]"
            out
        )

        fn toJsonPairArray names vals = (
            local out = "["
            local q = (bit.intAsChar 34)
            local lb = (bit.intAsChar 123)
            local rb = (bit.intAsChar 125)
            local lim = amin #(names.count, vals.count)
            for i = 1 to lim do (
                if i > 1 do out += ","
                out += lb + q + "name" + q + ":" + q + (esc names[i]) + q + "," + q + "value" + q + ":" + q + (esc vals[i]) + q + rb
            )
            out += "]"
            out
        )

        fn classifyDeclType decl = (
            local d = toLower decl
            if (findString d "texturemap") != undefined or (findString d "texmap") != undefined then "map"
            else if (findString d "color") != undefined then "color"
            else if (findString d "bool") != undefined then "bool"
            else if (findString d "float") != undefined or (findString d "integer") != undefined or (findString d "double") != undefined or (findString d "worldunits") != undefined or (findString d "percent") != undefined then "numeric"
            else "other"
        )

        local obj = getNodeByName "{safe}"
        if obj == undefined then (
            "{{\\"error\\":\\"Object not found: {safe}\\"}}"
        ) else if obj.material == undefined then (
            "{{\\"error\\":\\"No material assigned to {safe}\\"}}"
        ) else (
            local mat = {mat_expr}
            if mat == undefined then (
                "{{\\"error\\":\\"Sub-material index {sub_material_index} not found on {safe}\\"}}"
            ) else (
                local includeValues = {include_vals}
                local maxSlots = {max_slots}
                local subIdx = {sub_material_index}

                local props = #()
                try (props = makeUniqueArray (getPropNames mat)) catch ()

                -- Build declared type map from showProperties output
                local typeNames = #()
                local typeVals = #()
                try (
                    local ss = stringstream ""
                    showProperties mat to:ss
                    seek ss 0
                    while not (eof ss) do (
                        local ln = readline ss
                        local chunks = filterString ln ":"
                        if chunks.count >= 2 do (
                            local lhs = trimRight chunks[1]
                            local rhs = trimLeft chunks[2]
                            local lhsParts = filterString lhs ". "
                            if lhsParts.count >= 1 do (
                                local pnm = toLower lhsParts[lhsParts.count]
                                append typeNames pnm
                                append typeVals rhs
                            )
                        )
                    )
                ) catch ()

                fn getDeclType pname tNames tVals = (
                    local idx = findItem tNames (toLower pname)
                    if idx != 0 then tVals[idx] else ""
                )

                local mapNames = #();     local mapVals = #()
                local colorNames = #();   local colorVals = #()
                local numNames = #();     local numVals = #()
                local boolNames = #();    local boolVals = #()
                local otherNames = #();   local otherVals = #()

                local scanned = 0
                for p in props while scanned < maxSlots do (
                    local pname = p as string
                    if pname == "materialList" or pname == "maps" then continue

                    local val = undefined
                    local ok = true
                    try (val = getProperty mat p) catch (ok = false)
                    if not ok then continue

                    local decl = getDeclType pname typeNames typeVals
                    local cls = classifyDeclType decl
                    local rt = try ((classOf val) as string) catch "undefined"
                    local valStr = try (val as string) catch ""

                    if valStr.count > 120 do valStr = (substring valStr 1 120) + "..."

                    -- Fallback map detection for undeclared cases
                    local pnameL = toLower pname
                    if cls == "other" and ((matchPattern pnameL pattern:"*_map*" ignoreCase:true) or (matchPattern pnameL pattern:"*_shader*" ignoreCase:true) or ((findString (toLower rt) "texture") != undefined)) do cls = "map"

                    case cls of (
                        "map": (
                            append mapNames pname
                            append mapVals valStr
                        )
                        "color": (
                            append colorNames pname
                            append colorVals valStr
                        )
                        "numeric": (
                            append numNames pname
                            append numVals valStr
                        )
                        "bool": (
                            append boolNames pname
                            append boolVals valStr
                        )
                        default: (
                            append otherNames pname
                            append otherVals valStr
                        )
                    )
                    scanned += 1
                )

                local result = "{{"
                result += "\\"name\\":\\"" + (esc mat.name) + "\\","
                result += "\\"class\\":\\"" + (esc ((classOf mat) as string)) + "\\","
                result += "\\"subMaterialIndex\\":" + (subIdx as string) + ","
                result += "\\"inspectedCount\\":" + (scanned as string) + ","
                result += "\\"counts\\":{{"
                result += "\\"map\\":" + (mapNames.count as string) + ","
                result += "\\"color\\":" + (colorNames.count as string) + ","
                result += "\\"numeric\\":" + (numNames.count as string) + ","
                result += "\\"bool\\":" + (boolNames.count as string) + ","
                result += "\\"other\\":" + (otherNames.count as string)
                result += "}},"

                if includeValues then (
                    result += "\\"mapSlots\\":" + (toJsonPairArray mapNames mapVals) + ","
                    result += "\\"colorSlots\\":" + (toJsonPairArray colorNames colorVals) + ","
                    result += "\\"numericSlots\\":" + (toJsonPairArray numNames numVals) + ","
                    result += "\\"boolSlots\\":" + (toJsonPairArray boolNames boolVals) + ","
                    result += "\\"otherSlots\\":" + (toJsonPairArray otherNames otherVals)
                ) else (
                    result += "\\"mapSlots\\":" + (toJsonNameArray mapNames) + ","
                    result += "\\"colorSlots\\":" + (toJsonNameArray colorNames) + ","
                    result += "\\"numericSlots\\":" + (toJsonNameArray numNames) + ","
                    result += "\\"boolSlots\\":" + (toJsonNameArray boolNames) + ","
                    result += "\\"otherSlots\\":" + (toJsonNameArray otherNames)
                )

                result += "}}"
                result
            )
        )
    )"""
    response = client.send_command(maxscript, timeout=45.0)
    raw = response.get("result", "")
    if not raw:
        return raw

    try:
        payload = json.loads(raw)
    except Exception:
        return raw

    if not isinstance(payload, dict):
        return raw

    material_class = str(payload.get("class", ""))
    compact: dict[str, object] = {
        "name": payload.get("name", ""),
        "class": material_class,
        "subMaterialIndex": payload.get("subMaterialIndex", sub_material_index),
        "inspectedCount": payload.get("inspectedCount", 0),
        "counts": payload.get("counts", {}),
        "hints": _material_slot_hints(material_class),
    }

    if "error" in payload:
        compact = {
            "error": payload.get("error"),
            "hints": _material_slot_hints(material_class),
        }
        return json.dumps(compact, separators=(",", ":"))

    trunc: dict[str, int] = {}
    if slot_scope in {"map", "all"}:
        _truncate_slots(payload, "mapSlots", max_per_group, compact, trunc)
    if slot_scope == "all":
        _truncate_slots(payload, "colorSlots", max_per_group, compact, trunc)
        _truncate_slots(payload, "numericSlots", max_per_group, compact, trunc)
        _truncate_slots(payload, "boolSlots", max_per_group, compact, trunc)
        _truncate_slots(payload, "otherSlots", max_per_group, compact, trunc)

    if trunc:
        compact["truncatedFrom"] = trunc

    return json.dumps(compact, separators=(",", ":"))


@mcp.tool()
def create_texture_map(
    map_class: str,
    map_name: str = "",
    params: str = "",
    properties: dict[str, str] | None = None,
    global_var: str = "",
) -> str:
    """Create a texture map and store it as a MAXScript global variable."""
    if client.native_available:
        payload = {
            "map_class": map_class,
            "map_name": map_name,
            "params": params,
            "properties": properties or {},
            "global_var": global_var,
        }
        response = client.send_command(json.dumps(payload), cmd_type="native:create_texture_map")
        return response.get("result", "")

    safe_map_name = safe_string(map_name)
    name_param = f' name:"{safe_map_name}"' if map_name else ""

    # Generate global var name if not provided
    if not global_var:
        base = map_name if map_name else map_class
        # Clean to valid MAXScript identifier
        global_var = "".join(c if c.isalnum() or c == "_" else "_" for c in base)
        if global_var[0].isdigit():
            global_var = "m_" + global_var

    # Build property-setting lines
    prop_lines = ""
    if properties:
        lines = []
        for prop, val in properties.items():
            safe_prop = safe_string(prop)
            lines.append(
                f'try (global {global_var} ; {global_var}.{safe_prop} = {safe_value(val)}; '
                f'append okList "{safe_prop}") '
                f'catch (append errList ("{safe_prop}: " + (getCurrentException())))'
            )
        prop_lines = "\n            ".join(lines)

    maxscript = f"""(
        try (
            global {global_var} = {map_class}{name_param} {params}
            okList = #()
            errList = #()
            {"" if not prop_lines else prop_lines}
            msg = "Created " + (classof {global_var}) as string
            if {global_var}.name != undefined do msg += " \\\"" + {global_var}.name + "\\\""
            msg += " as global '{global_var}'"
            if okList.count > 0 do (
                msg += " | Set: "
                for i = 1 to okList.count do (if i > 1 do msg += ", "; msg += okList[i])
            )
            if errList.count > 0 do (
                msg += " | Errors: "
                for i = 1 to errList.count do (if i > 1 do msg += "; "; msg += errList[i])
            )
            msg
        ) catch (
            "Error: " + (getCurrentException())
        )
    )"""
    response = client.send_command(maxscript)
    return response.get("result", "")


@mcp.tool()
def set_texture_map_properties(
    global_var: str,
    properties: dict[str, str],
) -> str:
    """Set properties on a texture map stored as a MAXScript global variable."""
    if client.native_available:
        payload = json.dumps({"global_var": global_var, "properties": properties})
        response = client.send_command(payload, cmd_type="native:set_texture_map_properties")
        return response.get("result", "")

    lines = []
    for prop, val in properties.items():
        safe_prop = safe_string(prop)
        lines.append(
            f'try ({global_var}.{safe_prop} = {val}; append okList "{safe_prop}") '
            f'catch (append errList ("{safe_prop}: " + (getCurrentException())))'
        )
    set_block = "\n            ".join(lines)

    maxscript = f"""(
        try (
            global {global_var}
            if {global_var} == undefined then (
                "Error: global '{global_var}' not found"
            ) else (
                okList = #()
                errList = #()
                {set_block}
                msg = "Set " + (okList.count as string) + " properties on " + {global_var}.name
                if okList.count > 0 do (
                    msg += ": "
                    for i = 1 to okList.count do (if i > 1 do msg += ", "; msg += okList[i])
                )
                if errList.count > 0 do (
                    msg += " | Errors: "
                    for i = 1 to errList.count do (if i > 1 do msg += "; "; msg += errList[i])
                )
                msg
            )
        ) catch (
            "Error: " + (getCurrentException())
        )
    )"""
    response = client.send_command(maxscript)
    return response.get("result", "")


@mcp.tool()
def set_sub_material(
    name: str,
    sub_material_index: int,
    material_class: str = "",
    material_name: str = "",
    params: str = "",
    source_index: int = 0,
) -> str:
    """Create or assign a sub-material in a Multi/Sub-Object material slot."""
    if client.native_available:
        payload = {
            "name": name,
            "sub_material_index": sub_material_index,
            "material_class": material_class,
            "material_name": material_name,
            "params": params,
            "source_index": source_index,
        }
        response = client.send_command(json.dumps(payload), cmd_type="native:set_sub_material")
        return response.get("result", "")

    safe = safe_string(name)
    safe_mat_name = safe_string(material_name)
    name_param = f' name:"{safe_mat_name}"' if material_name else ""

    if source_index > 0:
        # Reference from another slot
        maxscript = f"""(
            obj = getNodeByName "{safe}"
            if obj == undefined then "Object not found: {safe}"
            else if obj.material == undefined then "No material on {safe}"
            else if (classof obj.material) != Multimaterial then "Material is not Multimaterial"
            else (
                try (
                    srcMat = obj.material.materialList[{source_index}]
                    if srcMat == undefined then "Source slot {source_index} is empty"
                    else (
                        obj.material.materialList[{sub_material_index}] = srcMat
                        "Sub[{sub_material_index}] = Sub[{source_index}] (" + srcMat.name + ") — shared reference"
                    )
                ) catch ("Error: " + (getCurrentException()))
            )
        )"""
    else:
        # Create new material at slot
        maxscript = f"""(
            obj = getNodeByName "{safe}"
            if obj == undefined then "Object not found: {safe}"
            else if obj.material == undefined then "No material on {safe}"
            else if (classof obj.material) != Multimaterial then "Material is not Multimaterial"
            else (
                try (
                    newMat = {material_class}{name_param} {params}
                    obj.material.materialList[{sub_material_index}] = newMat
                    "Sub[{sub_material_index}] = " + newMat.name + " (" + (classof newMat) as string + ")"
                ) catch ("Error: " + (getCurrentException()))
            )
        )"""
    response = client.send_command(maxscript)
    return response.get("result", "")


@mcp.tool()
def write_osl_shader(
    shader_name: str,
    osl_code: str,
    global_var: str = "",
    properties: dict[str, str] | None = None,
) -> str:
    """Write an OSL shader to disk and create an OSLMap from it."""
    if not global_var:
        global_var = "".join(c if c.isalnum() or c == "_" else "_" for c in shader_name)
        if global_var[0].isdigit():
            global_var = "m_" + global_var

    if client.native_available:
        payload = {
            "shader_name": shader_name,
            "osl_code": osl_code,
            "global_var": global_var,
        }
        if properties:
            payload["properties"] = properties
        response = client.send_command(json.dumps(payload), cmd_type="native:write_osl_shader")
        raw = response.get("result", "")
        try:
            data = json.loads(raw)
            return data.get("message", raw)
        except Exception:
            return raw

    # Escape the OSL code for MAXScript string embedding
    safe_osl = osl_code.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    safe_shader_name = safe_string(shader_name)

    # Build property-setting lines
    prop_lines = ""
    if properties:
        lines = []
        for prop, val in properties.items():
            safe_prop = safe_string(prop)
            lines.append(
                f'try (global {global_var} ; {global_var}.{safe_prop} = {safe_value(val)}; '
                f'append okList "{safe_prop}") '
                f'catch (append errList ("{safe_prop}: " + (getCurrentException())))'
            )
        prop_lines = "\n            ".join(lines)

    maxscript = f"""(
        try (
            oslDir = (getDir #temp) + "\\\\osl_shaders\\\\"
            makeDir oslDir
            oslPath = oslDir + "{safe_shader_name}.osl"
            oslContent = "{safe_osl}"
            f = createFile oslPath
            format "%" oslContent to:f
            close f

            global {global_var} = OSLMap name:"{safe_shader_name}"
            {global_var}.OSLCode = oslContent
            {global_var}.OSLAutoUpdate = true
            {global_var}.OSLPath = oslPath

            okList = #()
            errList = #()
            {"" if not prop_lines else prop_lines}

            msg = "OSL shader written to " + oslPath + " | Global: {global_var}"
            if okList.count > 0 do (
                msg += " | Set: "
                for i = 1 to okList.count do (if i > 1 do msg += ", "; msg += okList[i])
            )
            if errList.count > 0 do (
                msg += " | Errors: "
                for i = 1 to errList.count do (if i > 1 do msg += "; "; msg += errList[i])
            )
            msg
        ) catch (
            "Error: " + (getCurrentException())
        )
    )"""
    response = client.send_command(maxscript)
    return response.get("result", "")


@mcp.tool()
def create_material_from_textures(
    texture_folder: str,
    material_class: str = "",
    material_name: str = "",
    assign_to: StrList | None = None,
    custom_patterns: dict[str, list[str]] | None = None,
) -> str:
    """Create a fully-wired PBR material from a folder of texture maps."""
    # -- Step 1: Scan folder (Python-side) --
    files = _scan_texture_folder(texture_folder)
    if not files:
        return f"No image files found in: {texture_folder}"

    # -- Step 2: Match textures to channels (Python-side) --
    patterns = dict(_DEFAULT_CHANNEL_PATTERNS)
    if custom_patterns:
        patterns.update(custom_patterns)

    matched = _match_textures_to_channels(files, patterns)
    if not matched:
        suffixes = [f.stem for f in files[:10]]
        return f"No textures matched any channel pattern. File stems: {suffixes}"

    # -- Step 3: Determine renderer / material class --
    # Default is OpenPBR — neutral PBR material that works without a renderer.
    # Shell_Material is a wrapping construct (render slot + export slot), only
    # built when the caller explicitly asks for it via material_class.
    class_lower = material_class.lower().strip() if material_class else ""
    if not material_class:
        renderer = "openpbr"
    elif class_lower in {"shell", "shell_material", "shell_mtl", "shell_material_arnold", "shell_arnold_orm"}:
        renderer = "shell"
    elif "openpbr" in class_lower or "open_pbr" in class_lower:
        renderer = "openpbr"
    elif "ai_standard" in class_lower or "arnold" in class_lower:
        renderer = "arnold"
    elif "physical" in class_lower:
        renderer = "physical"
    elif "rs_standard" in class_lower or "redshift" in class_lower:
        renderer = "redshift"
    else:
        return (f"Unsupported material_class: {material_class}. "
                "Use OpenPBRMaterial, Shell_Material, ai_standard_surface, PhysicalMaterial, or RS_Standard_Material.")

    # -- Step 4: Derive material name --
    if not material_name:
        material_name = Path(texture_folder).name

    # -- Step 5: Build MAXScript --
    if renderer == "shell":
        if "diffuse" not in matched or "orm" not in matched:
            return "Shell_Material workflow requires at least diffuse/basecolor and packed ORM textures."
        maxscript = build_shell_maxscript(
            shell_name=material_name,
            render_name=f"{material_name}_arnold",
            base_color_path=matched["diffuse"],
            orm_path=matched["orm"],
            normal_path=matched.get("normal"),
            gltf_material_name=f"{material_name}_gltf",
            assign_to=assign_to,
            create_export_material=True,
        )
    elif renderer == "openpbr":
        maxscript = _build_openpbr_maxscript(matched, material_name, assign_to)
    elif renderer == "arnold":
        maxscript = _build_arnold_maxscript(matched, material_name, assign_to)
    elif renderer == "redshift":
        maxscript = _build_redshift_maxscript(matched, material_name, assign_to)
    else:
        maxscript = _build_physical_maxscript(matched, material_name, assign_to)

    # Wrap in try/catch
    maxscript = f"""(
    try (
        {maxscript}
    ) catch (
        "Error: " + (getCurrentException())
    )
)"""

    # -- Step 6: Send to Max --
    response = client.send_command(maxscript)
    return response.get("result", "")


def _scan_material_editor_palette_files(folder: str, recursive: bool) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []

    iterator = root.rglob("*") if recursive else root.iterdir()
    files = [
        path for path in iterator
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda path: str(path).lower())


def _build_material_editor_palette_maxscript(
    files: list[Path],
    start_slot: int,
    open_editor: bool,
    material_prefix: str,
    slot_content: str,
) -> str:
    lines: list[str] = [
        "fn mcp_setFirstMap target propNames tex = (",
        "    for propName in propNames do (",
        "        try (setProperty target (propName as name) tex; return propName) catch ()",
        "    )",
        "    undefined",
        ")",
        "fn mcp_setFirstValue target propNames value = (",
        "    for propName in propNames do (",
        "        try (setProperty target (propName as name) value; return propName) catch ()",
        "    )",
        "    undefined",
        ")",
        "fn mcp_createOpenPbrPreferred matName = (",
        "    local m = undefined",
        "    try (m = OpenPBRMaterial name:matName) catch ()",
        "    if m == undefined do try (m = OpenPBR_Material name:matName) catch ()",
        "    if m == undefined do try (m = OpenPBR_Mtl name:matName) catch ()",
        "    if m == undefined do try (m = PhysicalMaterial name:matName) catch ()",
        '    if m == undefined do throw "OpenPBRMaterial/OpenPBR_Material/OpenPBR_Mtl/PhysicalMaterial are unavailable"',
        "    m",
        ")",
        "local loaded = #()",
        "local classes = #()",
        "local errors = #()",
        f"local slotIndex = {start_slot}",
    ]

    if open_editor:
        lines.extend([
            "try (MatEditor.mode = #basic) catch ()",
            "try (MatEditor.Open()) catch ()",
        ])

    for idx, fpath in enumerate(files, start=1):
        path_literal = safe_string(_ms_path(fpath))
        tex_name = safe_string(fpath.stem)
        mat_name = safe_string(f"{material_prefix}{fpath.stem}")
        tex_var = f"tex_{idx}"
        mat_var = f"mat_{idx}"
        if slot_content == "bitmap":
            lines.extend([
                "try (",
                f'    local {tex_var} = Bitmaptexture name:"{tex_name}" filename:@"{path_literal}"',
                f"    try (medit.PutMtlToMtlEditor {tex_var} slotIndex) catch (meditMaterials[slotIndex] = {tex_var})",
                "    try (medit.SetActiveMtlSlot slotIndex true) catch (activeMeditSlot = slotIndex)",
                f'    append loaded ((slotIndex as string) + ": {safe_string(fpath.name)} -> " + ((classOf {tex_var}) as string))',
                f"    appendIfUnique classes ((classOf {tex_var}) as string)",
                "    slotIndex += 1",
                f') catch (append errors ("{safe_string(fpath.name)}: " + (getCurrentException())))',
            ])
        else:
            lines.extend([
                "try (",
                f'    local {tex_var} = Bitmaptexture name:"{tex_name}" filename:@"{path_literal}"',
                f'    local {mat_var} = mcp_createOpenPbrPreferred "{mat_name}"',
                f'    local slotName = mcp_setFirstMap {mat_var} #("base_color_map", "baseColor_map", "basecolor_map", "base_map", "diffuse_map") {tex_var}',
                f'    if slotName == undefined do try ({mat_var}.base_color_map = {tex_var}; slotName = "base_color_map") catch ()',
                f'    local specName = mcp_setFirstValue {mat_var} #("specular_color", "specularColor", "specular", "refl_color") (color 0 0 0)',
                f"    try (medit.PutMtlToMtlEditor {mat_var} slotIndex) catch (meditMaterials[slotIndex] = {mat_var})",
                "    try (medit.SetActiveMtlSlot slotIndex true) catch (activeMeditSlot = slotIndex)",
                f'    append loaded ((slotIndex as string) + ": {safe_string(fpath.name)} -> " + ((classOf {mat_var}) as string) + ", spec=" + (specName as string))',
                f"    appendIfUnique classes ((classOf {mat_var}) as string)",
                "    slotIndex += 1",
                f') catch (append errors ("{safe_string(fpath.name)}: " + (getCurrentException())))',
            ])

    content_label = "bitmap texture map" if slot_content == "bitmap" else "OpenPBR-first texture material"
    lines.extend([
        f'local msg = "Loaded " + (loaded.count as string) + " {content_label}(s) into Material Editor slots"',
        'if loaded.count > 0 do msg += " [" + loaded[1] + " .. " + loaded[loaded.count] + "]"',
        'if classes.count > 0 do msg += " | Classes: " + (classes as string)',
        'if errors.count > 0 do (',
        '    msg += " | Errors: "',
        '    for i = 1 to errors.count do (',
        '        if i > 1 do msg += "; "',
        '        msg += errors[i]',
        '    )',
        ')',
        "msg",
    ])
    return "(\n    " + "\n    ".join(lines) + "\n)"


def _build_material_editor_pbr_palette_maxscript(
    groups: list[dict],
    start_slot: int,
    open_editor: bool,
    material_prefix: str,
    renderer: str,
    include_displacement: bool = True,
    unmatched_count: int = 0,
    duplicate_count: int = 0,
) -> str:
    """Generate MAXScript for one fully wired PBR material per texture set."""
    renderer_label = {
        "openpbr": "OpenPBR-first",
        "materialx": "OpenPBR + MaterialX OSL",
        "physical": "PhysicalMaterial",
        "arnold": "Arnold ai_standard_surface",
        "redshift": "Redshift RS_Standard_Material",
        "vray": "V-Ray VRayMtl",
        "octane_standard": "Octane Std Surface (Std_Surface_Mtl)",
        "octane_pbr": "Octane Open PBR Surface (Open_PBR_Surf__Mtl)",
        "octane_universal": "Octane Universal (Universal_material)",
    }[renderer]
    is_octane = renderer.startswith("octane")

    lines: list[str] = [
        "fn mcp_setFirstMap target propNames tex = (",
        "    for propName in propNames do (",
        "        try (setProperty target (propName as name) tex; return propName) catch ()",
        "    )",
        "    undefined",
        ")",
        "fn mcp_setFirstValue target propNames value = (",
        "    for propName in propNames do (",
        "        try (setProperty target (propName as name) value; return propName) catch ()",
        "    )",
        "    undefined",
        ")",
        "fn mcp_enableMapSlot target slotName = (",
        "    if slotName != undefined do (",
        '        try (setProperty target ((slotName + "_on") as name) true) catch ()',
        '        try (setProperty target ((slotName + "_enable") as name) true) catch ()',
        '        try (',
        '            local s = slotName as string',
        '            if matchPattern s pattern:"*_tex" do (',
        '                local prefix = substring s 1 (s.count - 4)',
        '                setProperty target ((prefix + "_input_type") as name) 2',
        '            )',
        '        ) catch ()',
        "    )",
        ")",
        "fn mcp_createOpenPbrPreferred matName = (",
        "    local m = undefined",
        "    try (m = OpenPBRMaterial name:matName) catch ()",
        "    if m == undefined do try (m = OpenPBR_Material name:matName) catch ()",
        "    if m == undefined do try (m = OpenPBR_Mtl name:matName) catch ()",
        "    if m == undefined do try (m = PhysicalMaterial name:matName) catch ()",
        '    if m == undefined do throw "OpenPBRMaterial/OpenPBR_Material/OpenPBR_Mtl/PhysicalMaterial are unavailable"',
        "    m",
        ")",
        "local loaded = #()",
        "local classes = #()",
        "local errors = #()",
        f"local slotIndex = {start_slot}",
    ]

    if renderer == "materialx":
        lines.extend([
            "fn mcp_materialXOslRoot = (",
            "    local roots = #(",
            '        "C:\\\\ProgramData\\\\Autodesk\\\\ApplicationPlugins\\\\USD for 3ds Max 2027\\\\Contents\\\\MaterialX_plugin\\\\Contents\\\\OSL\\\\MaterialX",',
            '        "C:\\\\ProgramData\\\\Autodesk\\\\ApplicationPlugins\\\\USD for 3ds Max 2026\\\\Contents\\\\MaterialX_plugin\\\\Contents\\\\OSL\\\\MaterialX",',
            '        "C:\\\\ProgramData\\\\Autodesk\\\\ApplicationPlugins\\\\USD for 3ds Max 2025\\\\Contents\\\\MaterialX_plugin\\\\Contents\\\\OSL\\\\MaterialX"',
            "    )",
            '    for root in roots where doesFileExist (root + "\\\\tiledimage_color3.osl") do return root',
            '    throw "USD MaterialX OSL nodes are unavailable. Expected tiledimage_color3.osl under ProgramData Autodesk ApplicationPlugins."',
            ")",
            "fn mcp_materialXOslPath fileName = (",
            "    local path = (mcp_materialXOslRoot()) + \"\\\\\" + fileName",
            '    if not (doesFileExist path) do throw ("MaterialX OSL file is unavailable: " + path)',
            "    path",
            ")",
            "fn mcp_makeMaterialXOslMap fileName nodeName = (",
            "    local m = OSLMap()",
            "    m.name = nodeName",
            "    m.OSLPath = mcp_materialXOslPath fileName",
            "    m.OSLAutoUpdate = true",
            "    m",
            ")",
        ])

    if renderer != "materialx" and not is_octane and any("orm" in group["channels"] for group in groups):
        lines.append('local oslPath = (getDir #maxRoot) + "OSL\\\\UberBitmap2.osl"')

    if open_editor:
        lines.extend([
            "try (MatEditor.mode = #basic) catch ()",
            "try (MatEditor.Open()) catch ()",
        ])

    def add_wire(
        mat_var: str,
        slot_var: str,
        channel_label: str,
        tex_var: str,
        candidates: list[str],
    ) -> None:
        lines.extend([
            f"    local {slot_var} = mcp_setFirstMap {mat_var} {_ms_name_array(candidates)} {tex_var}",
            f"    mcp_enableMapSlot {mat_var} {slot_var}",
            f'    if {slot_var} != undefined then channelList += "{channel_label}->" + {slot_var} + ", " else skippedList += "{channel_label}, "',
        ])

    def add_bitmap(var: str, channel: str, fpath: Path) -> None:
        path_literal = safe_string(_ms_path(fpath))
        tex_name = safe_string(fpath.stem)
        if renderer == "materialx":
            is_color = channel in _COLOR_CHANNELS
            file_name = "tiledimage_vector3.osl" if channel == "normal" else (
                "tiledimage_color3.osl" if is_color else "tiledimage_float.osl"
            )
            colorspace = "srgb_texture" if is_color else ""
            lines.extend([
                f'    local {var} = mcp_makeMaterialXOslMap "{file_name}" "{tex_name}"',
                f'    {var}.file = @"{path_literal}"',
                f'    {var}.file_colorspace = "{colorspace}"',
            ])
            if channel == "normal":
                lines.append(f"    {var}.default1 = [0.5, 0.5, 1.0]")
        elif renderer == "arnold":
            color_space = "sRGB" if channel in _COLOR_CHANNELS else "Raw"
            lines.append(f'    local {var} = ai_image name:"{tex_name}" filename:@"{path_literal}" color_space:"{color_space}"')
        elif is_octane:
            # Leave colorSpace at Octane's default (_OctaneBuildIn_LINEAR_sRGB).
            # Other strings like "sRGB" or "_OctaneBuildIn_LINEAR" fall through to
            # OCIO lookup and error when no OCIO config is loaded. The material's
            # slot semantics (basecolor vs roughness vs metallic) drive how Octane
            # interprets the texture data, not this string.
            lines.append(f'    local {var} = Image_MTX()')
            lines.append(f'    {var}.name = "{tex_name}"')
            lines.append(f'    {var}.filename = @"{path_literal}"')
        else:
            lines.append(f'    local {var} = Bitmaptexture name:"{tex_name}" filename:@"{path_literal}"')

    def add_orm_split(prefix: str, fpath: Path) -> dict[str, str]:
        path_literal = safe_string(_ms_path(fpath))
        tex_name = safe_string(fpath.stem)
        uber = f"{prefix}_orm"
        out_r = f"{prefix}_orm_r"
        out_g = f"{prefix}_orm_g"
        out_b = f"{prefix}_orm_b"
        if renderer == "materialx":
            lines.extend([
                f'    local {uber} = mcp_makeMaterialXOslMap "tiledimage_color3.osl" "{tex_name}"',
                f'    {uber}.file = @"{path_literal}"',
                f'    {uber}.file_colorspace = ""',
                f'    local {out_r} = mcp_makeMaterialXOslMap "extract_color3.osl" "{tex_name}_AO_R"',
                f"    {out_r}.In_map = {uber}",
                f"    {out_r}.index = 0",
                f'    local {out_g} = mcp_makeMaterialXOslMap "extract_color3.osl" "{tex_name}_Roughness_G"',
                f"    {out_g}.In_map = {uber}",
                f"    {out_g}.index = 1",
                f'    local {out_b} = mcp_makeMaterialXOslMap "extract_color3.osl" "{tex_name}_Metallic_B"',
                f"    {out_b}.In_map = {uber}",
                f"    {out_b}.index = 2",
            ])
            return {"ao": out_r, "roughness": out_g, "metallic": out_b}
        if is_octane:
            lines.extend([
                f'    local {uber} = Image_MTX()',
                f'    {uber}.name = "{tex_name}"',
                f'    {uber}.filename = @"{path_literal}"',
                f'    local {out_r} = Channel_picker()',
                f'    {out_r}.name = "{tex_name}_AO_R"',
                f'    {out_r}.texture_tex = {uber}',
                f'    {out_r}.texture_input_type = 2',
                f'    {out_r}.colorChannel = {_OCTANE_CH_R}',
                f'    local {out_g} = Channel_picker()',
                f'    {out_g}.name = "{tex_name}_Rough_G"',
                f'    {out_g}.texture_tex = {uber}',
                f'    {out_g}.texture_input_type = 2',
                f'    {out_g}.colorChannel = {_OCTANE_CH_G}',
                f'    local {out_b} = Channel_picker()',
                f'    {out_b}.name = "{tex_name}_Metal_B"',
                f'    {out_b}.texture_tex = {uber}',
                f'    {out_b}.texture_input_type = 2',
                f'    {out_b}.colorChannel = {_OCTANE_CH_B}',
            ])
            return {"ao": out_r, "roughness": out_g, "metallic": out_b}
        lines.extend([
            f"    local {uber} = OSLMap()",
            f'    {uber}.name = "{tex_name}"',
            f"    {uber}.OSLPath = oslPath",
            f"    {uber}.OSLAutoUpdate = true",
            f'    {uber}.filename = @"{path_literal}"',
            f"    local {out_r} = MultiOutputChannelTexmapToTexmap()",
            f"    {out_r}.sourceMap = {uber}",
            f"    {out_r}.outputChannelIndex = {_UBER_OUT_R}",
            f"    local {out_g} = MultiOutputChannelTexmapToTexmap()",
            f"    {out_g}.sourceMap = {uber}",
            f"    {out_g}.outputChannelIndex = {_UBER_OUT_G}",
            f"    local {out_b} = MultiOutputChannelTexmapToTexmap()",
            f"    {out_b}.sourceMap = {uber}",
            f"    {out_b}.outputChannelIndex = {_UBER_OUT_B}",
        ])
        return {"ao": out_r, "roughness": out_g, "metallic": out_b}

    for idx, group in enumerate(groups, start=1):
        mat_name = safe_string(f"{material_prefix}{group['name']}")
        mat_var = f"mat_{idx}"
        channels: dict[str, Path] = group["channels"]
        map_vars: dict[str, str] = {}

        lines.append("try (")
        if renderer in {"openpbr", "materialx"}:
            lines.append(f'    local {mat_var} = mcp_createOpenPbrPreferred "{mat_name}"')
        elif renderer == "physical":
            lines.append(f'    local {mat_var} = PhysicalMaterial name:"{mat_name}"')
        elif renderer == "arnold":
            lines.append(f'    local {mat_var} = ai_standard_surface name:"{mat_name}"')
        elif renderer == "redshift":
            lines.append(f'    local {mat_var} = RS_Standard_Material name:"{mat_name}"')
        elif renderer == "octane_standard":
            lines.append(f'    local {mat_var} = Std_Surface_Mtl name:"{mat_name}"')
        elif renderer == "octane_pbr":
            lines.append(f'    local {mat_var} = Open_PBR_Surf__Mtl name:"{mat_name}"')
        elif renderer == "octane_universal":
            lines.append(f'    local {mat_var} = Universal_material name:"{mat_name}"')
        else:
            lines.append(f'    local {mat_var} = VRayMtl name:"{mat_name}"')
            lines.append(f"    try ({mat_var}.brdf_useRoughness = true) catch ()")

        lines.extend([
            "    local channelList = \"\"",
            "    local skippedList = \"\"",
            f'    local specDefault = mcp_setFirstValue {mat_var} #("specular_color", "specularColor", "specular", "refl_color", "reflection") (color 255 255 255)',
        ])

        for channel, fpath in channels.items():
            if channel == "orm":
                for split_channel, split_var in add_orm_split(f"g{idx}", fpath).items():
                    map_vars.setdefault(split_channel, split_var)
            else:
                var = f"g{idx}_{channel}"
                add_bitmap(var, channel, fpath)
                map_vars[channel] = var

        ao_var = map_vars.get("ao")

        slots = _PBR_SLOT_CANDIDATES[renderer]

        if "diffuse" in map_vars:
            diffuse_var = map_vars["diffuse"]
            if ao_var:
                if renderer == "materialx":
                    comp_var = f"g{idx}_diffuse_ao"
                    lines.extend([
                        f'    local {comp_var} = mcp_makeMaterialXOslMap "multiply_color3FA.osl" "Diffuse_AO"',
                        f"    {comp_var}.in1_map = {diffuse_var}",
                        f"    {comp_var}.in2_map = {ao_var}",
                    ])
                elif renderer == "arnold":
                    comp_var = f"g{idx}_diffuse_ao"
                    lines.extend([
                        f'    local {comp_var} = ai_multiply name:"Diffuse_AO"',
                        f"    {comp_var}.input1_shader = {diffuse_var}",
                        f"    {comp_var}.input2_shader = {ao_var}",
                    ])
                elif is_octane:
                    # Multiply_MTX collapses color to greyscale regardless of
                    # textureValueType. Multiply_texture preserves RGB.
                    comp_var = f"g{idx}_diffuse_ao"
                    lines.extend([
                        f'    local {comp_var} = Multiply_texture()',
                        f'    {comp_var}.name = "Diffuse_AO"',
                        f"    {comp_var}.texture1_tex = {diffuse_var}",
                        f"    {comp_var}.texture1_input_type = 2",
                        f"    {comp_var}.texture2_tex = {ao_var}",
                        f"    {comp_var}.texture2_input_type = 2",
                    ])
                else:
                    comp_var = f"g{idx}_diffuse_ao"
                    lines.extend([
                        f"    local {comp_var} = CompositeTexturemap()",
                        f'    {comp_var}.name = "Diffuse_AO"',
                        f"    {comp_var}.mapList[1] = {diffuse_var}",
                        f"    {comp_var}.mapList[2] = {ao_var}",
                        f"    {comp_var}.blendMode[2] = 5",
                    ])
                add_wire(mat_var, f"slot_{idx}_diffuse", "diffuse(+ao)", comp_var, slots["diffuse"])
            else:
                add_wire(mat_var, f"slot_{idx}_diffuse", "diffuse", diffuse_var, slots["diffuse"])
        elif "ao" in map_vars:
            lines.append('    skippedList += "ao(no diffuse), "')

        if "roughness" in map_vars:
            add_wire(mat_var, f"slot_{idx}_roughness", "roughness", map_vars["roughness"], slots["roughness"])
        elif "glossiness" in map_vars:
            inv_var = f"g{idx}_gloss_to_rough"
            if renderer == "materialx":
                lines.extend([
                    f'    local {inv_var} = mcp_makeMaterialXOslMap "invert_float.osl" "GlossToRough"',
                    f"    {inv_var}.In_map = {map_vars['glossiness']}",
                    f"    {inv_var}.amount = 1.0",
                ])
            elif is_octane:
                lines.extend([
                    f'    local {inv_var} = Invert_MTX()',
                    f'    {inv_var}.name = "GlossToRough"',
                    f"    {inv_var}.input_tex = {map_vars['glossiness']}",
                    f"    {inv_var}.input_input_type = 2",
                ])
            else:
                lines.extend([
                    f'    local {inv_var} = Output name:"GlossToRough"',
                    f"    {inv_var}.map1 = {map_vars['glossiness']}",
                    f"    {inv_var}.output.invert = true",
                ])
            add_wire(mat_var, f"slot_{idx}_glossiness", "glossiness(inverted)", inv_var, slots["glossiness"])
        elif "roughness" in map_vars:
            add_wire(mat_var, f"slot_{idx}_roughness_orm", "roughness(orm)", map_vars["roughness"], slots["roughness"])

        if "metallic" in map_vars:
            add_wire(mat_var, f"slot_{idx}_metallic", "metallic", map_vars["metallic"], slots["metallic"])

        if "normal" in map_vars:
            if renderer == "materialx":
                final_normal = f"g{idx}_normal_node"
                lines.extend([
                    f'    local {final_normal} = mcp_makeMaterialXOslMap "normalmap.osl" "NormalMap"',
                    f"    {final_normal}.In_map = {map_vars['normal']}",
                    f'    {final_normal}.space = "tangent"',
                    f"    {final_normal}.scale = 1.0",
                ])
            elif renderer == "arnold":
                normal_node = f"g{idx}_normal_node"
                final_normal = normal_node
                lines.extend([
                    f'    local {normal_node} = ai_normal_map name:"NormalMap" input_shader:{map_vars["normal"]}',
                ])
                if "bump" in map_vars:
                    bump_node = f"g{idx}_normal_bump_node"
                    lines.extend([
                        f'    local {bump_node} = ai_bump2d name:"NormalBump"',
                        f"    {bump_node}.bump_map_shader = {map_vars['bump']}",
                        f"    {bump_node}.normal_shader = {normal_node}",
                    ])
                    final_normal = bump_node
            elif renderer == "redshift":
                final_normal = f"g{idx}_normal_node"
                lines.extend([
                    f'    local {final_normal} = RS_BumpMap name:"NormalMap"',
                    f"    {final_normal}.input_map = {map_vars['normal']}",
                    f"    {final_normal}.inputType = 1",
                ])
            elif renderer == "vray":
                final_normal = f"g{idx}_normal_node"
                lines.extend([
                    f'    local {final_normal} = VRayNormalMap name:"NormalMap"',
                    f"    {final_normal}.normal_map = {map_vars['normal']}",
                ])
                if "bump" in map_vars:
                    lines.append(f"    {final_normal}.bump_map = {map_vars['bump']}")
            elif is_octane:
                # Octane materials have both normal_tex and bump_tex slots; wire each directly.
                # Bump is wired below in its own block when also present.
                final_normal = map_vars['normal']
            else:
                final_normal = f"g{idx}_normal_node"
                lines.extend([
                    f'    local {final_normal} = Normal_Bump name:"NormalBump"',
                    f"    {final_normal}.normal_map = {map_vars['normal']}",
                ])
                if "bump" in map_vars:
                    lines.append(f"    {final_normal}.bump_map = {map_vars['bump']}")
            add_wire(mat_var, f"slot_{idx}_normal", "normal", final_normal, slots["normal"])
            if is_octane and "bump" in map_vars:
                add_wire(mat_var, f"slot_{idx}_bump", "bump", map_vars["bump"], slots["bump"])
        elif "bump" in map_vars:
            if renderer == "materialx":
                height_node = f"g{idx}_height_to_normal"
                bump_node = f"g{idx}_bump_node"
                lines.extend([
                    f'    local {height_node} = mcp_makeMaterialXOslMap "heighttonormal_vector3.osl" "BumpHeightToNormal"',
                    f"    {height_node}.In_map = {map_vars['bump']}",
                    f"    {height_node}.scale = 1.0",
                    f'    local {bump_node} = mcp_makeMaterialXOslMap "normalmap.osl" "BumpNormalMap"',
                    f"    {bump_node}.In_map = {height_node}",
                    f'    {bump_node}.space = "tangent"',
                    f"    {bump_node}.scale = 1.0",
                ])
            elif renderer == "arnold":
                bump_node = f"g{idx}_bump_node"
                lines.extend([
                    f'    local {bump_node} = ai_bump2d name:"Bump"',
                    f"    {bump_node}.bump_map_shader = {map_vars['bump']}",
                ])
            elif renderer == "redshift":
                bump_node = f"g{idx}_bump_node"
                lines.extend([
                    f'    local {bump_node} = RS_BumpMap name:"Bump"',
                    f"    {bump_node}.input_map = {map_vars['bump']}",
                    f"    {bump_node}.inputType = 0",
                ])
            elif renderer == "vray":
                bump_node = map_vars["bump"]
            elif is_octane:
                bump_node = map_vars["bump"]
            else:
                bump_node = f"g{idx}_bump_node"
                lines.extend([
                    f'    local {bump_node} = Normal_Bump name:"Bump"',
                    f"    {bump_node}.bump_map = {map_vars['bump']}",
                ])
            add_wire(mat_var, f"slot_{idx}_bump", "bump", bump_node, slots["bump"])

        optional_channels = ("displacement", "opacity", "emission", "translucency", "specular")
        if not include_displacement:
            optional_channels = ("opacity", "emission", "translucency", "specular")

        for channel in optional_channels:
            if channel not in map_vars:
                continue
            candidates = slots.get(channel)
            if not candidates:
                lines.append(f'    skippedList += "{channel}, "')
                continue
            wire_var = map_vars[channel]
            if channel == "displacement" and is_octane:
                # Octane's displacement slot rejects raw Image_MTX; wrap in Texture_displacement.
                td_var = f"g{idx}_disp_node"
                lines.extend([
                    f'    local {td_var} = Texture_displacement()',
                    f'    {td_var}.name = "Displacement"',
                    f'    {td_var}.texture_tex = {wire_var}',
                    f'    {td_var}.texture_input_type = 2',
                ])
                wire_var = td_var
            add_wire(mat_var, f"slot_{idx}_{channel}", channel, wire_var, candidates)

        if "ior" in map_vars:
            lines.append('    skippedList += "ior(no map slot), "')

        lines.extend([
            f"    try (medit.PutMtlToMtlEditor {mat_var} slotIndex) catch (meditMaterials[slotIndex] = {mat_var})",
            "    try (medit.SetActiveMtlSlot slotIndex true) catch (activeMeditSlot = slotIndex)",
            f'    append loaded ((slotIndex as string) + ": " + {mat_var}.name + " [" + channelList + "]")',
            f"    appendIfUnique classes ((classOf {mat_var}) as string)",
            "    slotIndex += 1",
            f') catch (append errors ("{mat_name}: " + (getCurrentException())))',
        ])

    lines.extend([
        f'local msg = "Loaded " + (loaded.count as string) + " grouped PBR material(s) into Material Editor slots using {renderer_label}"',
        'if loaded.count > 0 do msg += " [" + loaded[1] + " .. " + loaded[loaded.count] + "]"',
        'if classes.count > 0 do msg += " | Classes: " + (classes as string)',
        f'if {unmatched_count} > 0 do msg += " | Unmatched image(s) skipped: {unmatched_count}"',
        f'if {duplicate_count} > 0 do msg += " | Duplicate channel file(s) skipped: {duplicate_count}"',
        'if errors.count > 0 do (',
        '    msg += " | Errors: "',
        '    for i = 1 to errors.count do (',
        '        if i > 1 do msg += "; "',
        '        msg += errors[i]',
        '    )',
        ')',
        "msg",
    ])
    return "(\n    " + "\n    ".join(lines) + "\n)"


def _palette_laydown_impl(
    texture_folder: str,
    start_slot: int = 1,
    max_slots: int = 24,
    recursive: bool = False,
    open_editor: bool = True,
    material_prefix: str = "tex_",
    slot_content: str = "material",
    material_class: str = "",
    include_displacement: bool = True,
) -> str:
    """Load image files from a folder into Compact Material Editor sample slots.

    slot_content="material" creates OpenPBR-first preview materials, wires each
    bitmap into base color, and sets specular color to black. slot_content="bitmap"
    places raw Bitmaptexture maps directly into the palette slots. slot_content
    values like "pbr_material" or "full_pbr" group texture sets by filename and
    create one fully wired PBR material per slot. For grouped mode, material_class
    may be OpenPBRMaterial, PhysicalMaterial, ai_standard_surface,
    RS_Standard_Material, VRayMtl, MaterialX, Std_Surface_Mtl (octane_standard),
    Open_PBR_Surf__Mtl (octane_pbr), or Universal_material (octane_universal);
    OpenPBR is the default. Octane variants build with Image_MTX, Channel_picker
    (for packed ORM), Multiply_MTX (diffuse x AO), and Invert_MTX (gloss to
    roughness), and flip each material slot's *_input_type to 2 so the texture
    actually drives the channel. include_displacement=False skips wiring
    height/displacement maps in grouped PBR mode.
    """
    start_slot = max(1, min(24, int(start_slot)))
    max_slots = max(1, min(24 - start_slot + 1, int(max_slots)))
    raw_slot_content = (slot_content or "material").strip().lower()
    if raw_slot_content in {"material", "materials", "openpbr", "openpbr_material"}:
        slot_content = "material"
    elif raw_slot_content in {"bitmap", "bitmaps", "map", "maps", "texture", "textures"}:
        slot_content = "bitmap"
    elif raw_slot_content in {
        "pbr", "pbr_material", "pbr_materials", "full_pbr", "full_pbr_material",
        "grouped", "grouped_material", "grouped_materials", "renderer_material",
        "renderer_materials",
    }:
        slot_content = "pbr_material"
    else:
        return (
            f"Unsupported slot_content: {slot_content}. "
            "Use 'material' for OpenPBR preview materials, 'bitmap' for raw Bitmaptexture maps, "
            "or 'pbr_material' for grouped full PBR materials."
        )

    renderer = _renderer_from_material_class(material_class)
    if slot_content == "pbr_material" and renderer is None:
        return (
            f"Unsupported material_class for grouped PBR palette: {material_class}. "
            "Use OpenPBRMaterial, PhysicalMaterial, ai_standard_surface, RS_Standard_Material, "
            "VRayMtl, MaterialX, Std_Surface_Mtl (octane_standard), Open_PBR_Surf__Mtl (octane_pbr), "
            "or Universal_material (octane_universal)."
        )

    files = _scan_material_editor_palette_files(texture_folder, recursive)
    if not files:
        return f"No image files found in: {texture_folder}"

    if slot_content == "pbr_material":
        groups, unmatched, duplicates = _group_texture_files_for_pbr(files, _DEFAULT_CHANNEL_PATTERNS)
        if not groups:
            stems = [f.stem for f in files[:10]]
            return f"No texture sets matched any PBR channel pattern. File stems: {stems}"

        selected_groups = groups[:max_slots]
        maxscript = f"""(
    try (
        {_build_material_editor_pbr_palette_maxscript(
            selected_groups,
            start_slot,
            open_editor,
            material_prefix,
            renderer or "openpbr",
            include_displacement=include_displacement,
            unmatched_count=len(unmatched),
            duplicate_count=len(duplicates),
        )}
    ) catch (
        "Error: " + (getCurrentException())
    )
)"""
        response = client.send_command(maxscript)
        return response.get("result", "")

    selected = files[:max_slots]
    maxscript = f"""(
    try (
        {_build_material_editor_palette_maxscript(selected, start_slot, open_editor, material_prefix, slot_content)}
    ) catch (
        "Error: " + (getCurrentException())
    )
)"""
    response = client.send_command(maxscript)
    return response.get("result", "")


@mcp.tool()
def create_shell_material(
    shell_name: str,
    render_material_name: str,
    base_color_path: str,
    orm_path: str,
    normal_path: str = "",
    gltf_material_name: str = "",
    assign_to: StrList | None = None,
) -> str:
    """Create a Shell Material with UberBitmap-based Arnold render slot and glTF export slot."""
    if client.native_available and gltf_material_name:
        try:
            payload = json.dumps({
                "name": shell_name,
                "render_material_name": render_material_name,
                "base_color_path": base_color_path,
                "orm_path": orm_path,
                "normal_path": normal_path,
                "gltf_material_name": gltf_material_name,
                "assign_to": assign_to or [],
            })
            response = client.send_command(payload, cmd_type="native:create_shell_material")
            return response.get("result", "{}")
        except RuntimeError:
            pass

    maxscript = build_shell_maxscript(
        shell_name=shell_name,
        render_name=render_material_name,
        base_color_path=base_color_path,
        orm_path=orm_path,
        normal_path=normal_path or None,
        gltf_material_name=gltf_material_name or None,
        assign_to=assign_to,
        create_export_material=not bool(gltf_material_name),
    )

    maxscript = f"""(
    try (
        {maxscript}
    ) catch (
        "{{\\"status\\":\\"error\\",\\"error\\":\\"" + (getCurrentException()) + "\\"}}"
    )
)"""

    response = client.send_command(maxscript)
    return response.get("result", "{}")
