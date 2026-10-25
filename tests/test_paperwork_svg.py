import io
import unittest
from pathlib import Path

from PIL import Image

from dataforge.paperwork.svg2png import svg2png


class SvgRenderTests(unittest.TestCase):
    def _image(self, svg):
        return Image.open(io.BytesIO(svg2png(svg, scale=1))).convert('RGB')

    def test_explicit_times_new_roman_uses_real_font(self):
        import dataforge.paperwork.svg2png as svg2png
        font = svg2png._font(20, 'Times New Roman')
        self.assertEqual(font.getname()[0], 'Times New Roman')

    def test_text_middle_anchor_is_centered(self):
        image = self._image('<svg width="100" height="40"><text x="50" y="28" text-anchor="middle" font-size="20">WIDE</text></svg>')
        dark = [(x, y) for y in range(image.height) for x in range(image.width)
                if image.getpixel((x, y)) != (255, 255, 255)]
        self.assertTrue(dark)
        xs = [x for x, _ in dark]
        self.assertAlmostEqual((min(xs) + max(xs)) / 2, 50, delta=2)

    def test_svg_primitives_dash_marker_entities_and_rotated_text(self):
        svg = '''<svg width="80" height="80">
        <rect width="80" height="80" fill="white"/>
        <line x1="5" y1="10" x2="35" y2="10" stroke="#333333" stroke-width="2" stroke-dasharray="4,2"/>
        <polyline points="5,20 35,20 35,35" fill="none" stroke="#1f77b4" stroke-width="2"/>
        <circle cx="35" cy="35" r="3" fill="#d62728"/>
        <text x="60" y="50" text-anchor="middle" transform="rotate(-90 60 50)" font-size="12">A&amp;B</text>
        </svg>'''
        image = self._image(svg)
        self.assertNotEqual(image.getpixel((35, 35)), (255, 255, 255))
        self.assertNotEqual(image.getpixel((7, 10)), (255, 255, 255))
        self.assertEqual(image.getpixel((10, 10)), (255, 255, 255))
        self.assertNotEqual(image.getpixel((60, 45)), (255, 255, 255))

    def test_explicit_none_dasharray_renders_solid_line(self):
        image = self._image('<svg width="40" height="20"><line x1="2" y1="10" x2="35" y2="10" stroke="black" stroke-dasharray="none"/></svg>')
        self.assertTrue(all(image.getpixel((x, 10)) == (0, 0, 0) for x in range(3, 35)))

    def test_unknown_required_element_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'unsupported SVG element'):
            svg2png('<svg width="10" height="10"><path d="M0 0"/></svg>')


if __name__ == '__main__':
    unittest.main()
