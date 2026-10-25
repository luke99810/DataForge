"""Rasterize the small SVG subset emitted by Idea2Paper."""
import io
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NAMED_COLORS = {
    'white': (255, 255, 255), 'black': (0, 0, 0), 'none': None,
    '#1f77b4': (31, 119, 180), '#d62728': (214, 39, 40),
    '#dddddd': (221, 221, 221), '#eeeeee': (238, 238, 238),
    '#333333': (51, 51, 51),
}


def parse_color(value):
    value = value.strip().lower()
    if value.startswith('#') and len(value) == 7:
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
    if value.startswith('rgb('):
        return tuple(int(x.strip()) for x in value[4:-1].split(','))
    color = NAMED_COLORS.get(value)
    if color is None and value not in ('none',):
        raise ValueError('unsupported SVG color: ' + value)
    return color


def _font(size, family):
    if family.strip().strip('"\'').lower() == 'times new roman':
        for path in [Path('C:/Windows/Fonts/times.ttf'), Path.home() / '.fonts/times.ttf']:
            if path.is_file():
                return ImageFont.truetype(str(path), size)
        raise ValueError('Times New Roman is required but unavailable; font substitution forbidden')
    candidates = []
    if family:
        candidates.extend(name.strip().strip('"\'') for name in family.split(','))
    candidates += ['Arial', 'DejaVuSans']
    roots = [Path('C:/Windows/Fonts'), Path('/usr/share/fonts/truetype/dejavu')]
    for name in candidates:
        names = [name, name + '.ttf'] if name.lower().endswith('.ttf') else [name + '.ttf', name]
        for root in roots:
            for filename in names:
                path = root / filename
                if path.exists():
                    return ImageFont.truetype(str(path), size)
    raise ValueError('no supported TrueType font available for SVG text')


def _attrs(element):
    return {key.rsplit('}', 1)[-1]: value for key, value in element.attrib.items()}


def _points(value):
    return [(float(x), float(y)) for x, y in zip(
        re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', value)[::2],
        re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', value)[1::2])]


def svg2png(svg_text, scale=2):
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError('invalid SVG: ' + str(exc)) from exc
    root_attrs = _attrs(root)
    try:
        width, height = float(root_attrs['width']), float(root_attrs['height'])
    except (KeyError, ValueError) as exc:
        raise ValueError('SVG width and height are required') from exc
    image = Image.new('RGB', (round(width * scale), round(height * scale)), 'white')
    draw = ImageDraw.Draw(image)

    def xy(value):
        return round(float(value) * scale)

    def stroke(attrs):
        color = parse_color(attrs.get('stroke', 'black'))
        if color is None:
            return None, 1
        return color, max(1, round(float(attrs.get('stroke-width', 1)) * scale))

    def dashed_line(points, color, width, dash):
        if not dash or dash.strip().lower() == 'none':
            draw.line([(xy(x), xy(y)) for x, y in points], fill=color, width=width, joint='curve')
            return
        pattern = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', dash)]
        if not pattern or any(x <= 0 for x in pattern):
            raise ValueError('unsupported SVG stroke-dasharray: ' + dash)
        if len(pattern) % 2:
            pattern *= 2
        pattern = [x * scale for x in pattern]
        distance = 0.0
        on = True
        index = 0
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            length = math.hypot(x1 - x0, y1 - y0) * scale
            if not length:
                continue
            ux, uy = (x1 - x0) / length * scale, (y1 - y0) / length * scale
            consumed = 0.0
            while consumed < length:
                take = min(pattern[index] - distance, length - consumed)
                if on:
                    start = (x0 + ux * consumed / scale, y0 + uy * consumed / scale)
                    end = (x0 + ux * (consumed + take) / scale, y0 + uy * (consumed + take) / scale)
                    draw.line([(xy(start[0]), xy(start[1])), (xy(end[0]), xy(end[1]))], fill=color, width=width)
                consumed += take
                distance += take
                if distance >= pattern[index] - 1e-9:
                    distance = 0.0
                    index = (index + 1) % len(pattern)
                    on = not on

    def text_layer(attrs, body):
        x, y = float(attrs.get('x', 0)), float(attrs.get('y', 0))
        size = float(attrs.get('font-size', 12)) * scale
        font = _font(round(size), attrs.get('font-family', ''))
        color = parse_color(attrs.get('fill', 'black'))
        if color is None:
            return
        bbox = font.getbbox(body)
        text_width = draw.textlength(body, font=font)
        anchor = attrs.get('text-anchor', 'start')
        if anchor == 'middle':
            x -= text_width / scale / 2
        elif anchor == 'end':
            x -= text_width / scale
        elif anchor != 'start':
            raise ValueError('unsupported text-anchor: ' + anchor)
        layer = Image.new('RGBA', image.size, (255, 255, 255, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text((xy(x), xy(y) - bbox[3]), body, font=font, fill=color + (255,))
        transform = attrs.get('transform', '')
        match = re.fullmatch(r'\s*rotate\(\s*([-+.\d]+)(?:\s+([-+.\d]+)\s+([-+.\d]+))?\s*\)\s*', transform)
        if transform and not match:
            raise ValueError('unsupported SVG transform: ' + transform)
        if match:
            angle = float(match.group(1))
            cx = xy(match.group(2) or x)
            cy = xy(match.group(3) or y)
            layer = layer.rotate(angle, resample=Image.Resampling.BICUBIC, center=(cx, cy), fillcolor=(255, 255, 255, 0))
        image.paste(layer.convert('RGB'), (0, 0), layer.getchannel('A'))

    supported = {'rect', 'line', 'polyline', 'polygon', 'text', 'circle'}
    for element in root.iter():
        tag = element.tag.rsplit('}', 1)[-1]
        if element is root:
            continue
        if tag not in supported:
            raise ValueError('unsupported SVG element: ' + tag)
        attrs, body = _attrs(element), element.text or ''
        if tag == 'rect':
            color = parse_color(attrs.get('fill', 'black'))
            if color is not None:
                draw.rectangle((xy(attrs.get('x', 0)), xy(attrs.get('y', 0)),
                                xy(float(attrs.get('x', 0)) + float(attrs.get('width', 0))),
                                xy(float(attrs.get('y', 0)) + float(attrs.get('height', 0)))), fill=color)
        elif tag == 'circle':
            color = parse_color(attrs.get('fill', attrs.get('stroke', 'black')))
            if color is not None:
                cx, cy, radius = xy(attrs['cx']), xy(attrs['cy']), xy(attrs['r'])
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
        elif tag in ('line', 'polyline', 'polygon'):
            points = ([(float(attrs['x1']), float(attrs['y1'])), (float(attrs['x2']), float(attrs['y2']))]
                      if tag == 'line' else _points(attrs.get('points', '')))
            if tag == 'polygon' and points:
                points.append(points[0])
            color, width = stroke(attrs)
            if color is not None and len(points) >= 2:
                dashed_line(points, color, width, attrs.get('stroke-dasharray'))
        else:
            text_layer(attrs, body)

    raw = bytearray()
    pixels = image.load()
    for y in range(image.height):
        raw.append(0)
        for x in range(image.width):
            raw.extend(pixels[x, y])
    return _png_encode(image.width, image.height, bytes(raw))


def _png_encode(w, h, raw):
    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    return sig + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b'')


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    scale = int(args[3]) if len(args) > 3 and args[2] == '--scale' else 2
    svg = Path(args[0]).read_text(encoding='utf-8')
    out = Path(args[1])
    out.write_bytes(svg2png(svg, scale))
    print({'status': 'ok', 'png': str(out)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
