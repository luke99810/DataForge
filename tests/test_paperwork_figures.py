"""Regression checks for the demonstration's publication figures."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1] / 'dataforge' / 'paperwork'
spec = importlib.util.spec_from_file_location('logistics', ROOT / 'examples/logistics-demo/code/logistics.py')
logistics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logistics)


class ChartTests(unittest.TestCase):
    def test_chart_orders_sensitivity_points_by_x(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.svg'
            logistics.make_chart([0, 5, 2, 8], {'S2': [98, 1, 10, 0]}, path, 'Window', 'On-time (%)', 100)
            curve = ET.fromstring(path.read_text()).find('{*}polyline')
            xs = [float(point.split(',')[0]) for point in curve.get('points').split()]
            self.assertEqual(xs, sorted(xs))

    def test_accessible_series_and_legends_inside_canvas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.svg'
            logistics.make_chart([0.2, 0.5, 1.0], {'S1': [122, 315, 596], 'S2': [39, 69, 89]}, path, 'Batches', 'Vehicle batches', 660)
            svg = ET.fromstring(path.read_text(encoding='utf-8'))
            xml = path.read_text(encoding='utf-8')
            self.assertTrue(all(t.get('font-family') == 'Times New Roman' for t in svg.findall('{*}text')))
            self.assertTrue(all((t.text or '').isascii() for t in svg.findall('{*}text')))
            self.assertNotIn('#d62728', xml, 'Red series violates the skill palette')
            self.assertIn('#0072B2', xml)
            self.assertIn('#009E73', xml)
            curves = svg.findall('{*}polyline')
            self.assertNotEqual(curves[0].get('stroke-dasharray'), curves[1].get('stroke-dasharray'))
            width = float(svg.get('width'))
            self.assertTrue(all(float(t.get('x', 0)) < width - 20 for t in svg.findall('{*}text')))


if __name__ == '__main__':
    unittest.main()
