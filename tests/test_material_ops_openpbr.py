import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tools.material_ops import create_material_from_textures


class OpenPBRMaterialTests(unittest.TestCase):
    def test_create_material_from_textures_defaults_to_openpbr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tex = Path(tmp) / "asset_basecolor.png"
            tex.write_bytes(b"fake")

            with (
                patch(
                    "src.tools.material_ops._build_openpbr_maxscript",
                    return_value='("openpbr")',
                ) as build_openpbr,
                patch("src.tools.material_ops.client.send_command", return_value={"result": "ok"}) as send,
            ):
                result = create_material_from_textures(tmp)

        build_openpbr.assert_called_once()
        send.assert_called_once()
        self.assertEqual(result, "ok")

    def test_create_material_from_textures_accepts_explicit_openpbr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tex = Path(tmp) / "asset_roughness.png"
            tex.write_bytes(b"fake")

            with patch(
                "src.tools.material_ops._build_openpbr_maxscript",
                return_value='("openpbr")',
            ) as build_openpbr, patch(
                "src.tools.material_ops.client.send_command",
                return_value={"result": "ok"},
            ):
                create_material_from_textures(tmp, material_class="OpenPBR_Material")

        build_openpbr.assert_called_once()

    def test_create_material_from_textures_defaults_to_openpbr_even_with_orm(self) -> None:
        # Shell is a wrapping construct (render slot + export slot) and is
        # never the implicit default. Folders with packed ORM still get plain
        # OpenPBR unless the caller passes material_class="Shell_Material".
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "asset_basecolor.png").write_bytes(b"fake")
            (root / "asset_orm.png").write_bytes(b"fake")

            with patch("src.tools.material_ops._build_openpbr_maxscript", return_value="-- mock") as build_openpbr, \
                 patch("src.tools.material_ops.client.send_command",
                       return_value={"result": '{"status":"success"}'}):
                create_material_from_textures(tmp, material_name="asset")

        build_openpbr.assert_called_once()

    def test_create_material_from_textures_uses_shell_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "asset_basecolor.png").write_bytes(b"fake")
            (root / "asset_orm.png").write_bytes(b"fake")
            (root / "asset_normal.png").write_bytes(b"fake")

            with patch(
                "src.tools.material_ops.client.send_command",
                return_value={"result": '{"status":"success"}'},
            ) as send:
                result = create_material_from_textures(
                    tmp, material_name="asset", material_class="Shell_Material"
                )

        self.assertEqual(result, '{"status":"success"}')
        send.assert_called_once()
        maxscript = send.call_args.args[0]
        self.assertIn("Shell_Material()", maxscript)
        self.assertIn('OSL\\\\UberBitmap2.osl', maxscript)
        self.assertIn("MultiOutputChannelTexmapToTexmap", maxscript)
        self.assertIn("outputChannelIndex = 2", maxscript)
        self.assertIn("outputChannelIndex = 3", maxscript)
        self.assertIn("outputChannelIndex = 4", maxscript)
        self.assertIn("ai_multiply", maxscript)
        self.assertIn('shell.originalMaterial = arnoldMat', maxscript)
        self.assertIn('shell.bakedMaterial = gltfMat', maxscript)

    def test_create_material_from_textures_shell_requires_orm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "asset_basecolor.png").write_bytes(b"fake")

            with patch("src.tools.material_ops.client.send_command") as send:
                result = create_material_from_textures(tmp, material_class="Shell_Material")

        self.assertIn("requires at least diffuse/basecolor and packed ORM", result)
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
