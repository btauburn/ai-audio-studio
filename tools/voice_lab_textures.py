"""Procedural textures for the Voice Lab page: the liquid inside the glass buttons and the velvet (or cloth) behind everything.

Nothing here runs in Studio. It renders the images once and writes them into ``voice-lab.css`` as data URIs
between the ``/* textures:begin */`` and ``/* textures:end */`` markers (the page's private route serves only its
CSS and JS, so the images travel inside the stylesheet).

    python tools/voice_lab_textures.py              # regenerate and write into voice-lab.css
    python tools/voice_lab_textures.py --preview D  # also save PNG previews into folder D

Needs numpy and Pillow (with WebP). Every texture is seeded, so the output is the same each run.
"""
from __future__ import annotations

import argparse
import base64
import io
import pathlib
import re

import numpy as np
from PIL import Image

CSS = pathlib.Path(__file__).resolve().parents[1] / "dubbing_service" / "webui" / "voice-lab.css"

# ------------------------------------------------------------------------------------------------ noise helpers
def periodic_noise(shape, cycles_x, cycles_y, rng):
    """Smooth, tileable noise: white noise low-pass filtered in the frequency domain (features ~ size/cycles)."""
    h, w = shape
    white = rng.standard_normal(shape)
    fy = np.fft.fftfreq(h)[:, None] * h
    fx = np.fft.fftfreq(w)[None, :] * w
    kernel = np.exp(-((fx / cycles_x) ** 2 + (fy / cycles_y) ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(white) * kernel))
    return (out - out.mean()) / (out.std() + 1e-9)


def fbm(shape, base_x, base_y, rng, octaves=4, gain=0.5):
    total, amp, norm = np.zeros(shape), 1.0, 0.0
    for octave in range(octaves):
        total += amp * periodic_noise(shape, base_x * 2 ** octave, base_y * 2 ** octave, rng)
        norm += amp
        amp *= gain
    return total / norm


def sample(field, u, v):
    """Bilinear lookup in a periodic grid; u, v are in grid cells."""
    h, w = field.shape
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = u - x0, v - y0
    x0, y0 = x0 % w, y0 % h
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x1] * fx * (1 - fy)
            + field[y1, x0] * (1 - fx) * fy + field[y1, x1] * fx * fy)


def blur(img, radius):
    """Three box blurs ~ a Gaussian, along both axes, on an (H, W) or (H, W, C) array."""
    out = img.astype(np.float64)
    r = max(1, int(radius))
    for axis in (0, 1):
        for _ in range(3):
            pad = [(0, 0)] * out.ndim
            pad[axis] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
            n = out.shape[axis]
            out = (np.take(c, range(2 * r + 1, 2 * r + 1 + n), axis=axis) - np.take(c, range(0, n), axis=axis)) / (2 * r + 1)
    return out


def palette_at(stops, t):
    stops = np.asarray(stops, dtype=np.float64) / 255.0
    n = len(stops)
    pos = (t % 1.0) * n
    i0 = np.floor(pos).astype(int) % n
    i1 = (i0 + 1) % n
    f = (pos - np.floor(pos))[..., None]
    f = f * f * (3 - 2 * f)
    return stops[i0] * (1 - f) + stops[i1] * f


# ------------------------------------------------------------------------------------------------ liquid in glass
# Each family's liquid: a deep tone it sinks to, and the colours that flow through it. From the reference image:
# "Create" is magenta, orange and gold; "Read" is cyan, blue and violet; "Organize" is champagne, rose and lavender.
LIQUIDS = {
    "ember": {"deep": (70, 8, 46), "stops": [(255, 52, 150), (255, 104, 60), (255, 170, 52), (255, 72, 170), (196, 60, 255), (255, 132, 74)]},
    "aurora": {"deep": (20, 14, 78), "stops": [(52, 222, 255), (36, 214, 190), (76, 132, 255), (146, 92, 255), (222, 88, 255), (64, 190, 255)]},
    "pearl": {"deep": (74, 52, 82), "stops": [(236, 182, 112), (214, 150, 196), (176, 140, 232), (240, 196, 150), (226, 140, 176), (190, 160, 236)]},
    "mint": {"deep": (6, 58, 68), "stops": [(64, 236, 176), (52, 204, 255), (124, 150, 255), (140, 246, 168), (58, 226, 210)]},
}


def liquid(name, width=400, height=200, seed=7):
    """A seamless tile of iridescent liquid: marbled streams of colour flowing through a deep tone, with bright
    streaks and glitter. It tiles in both directions, so the page can slide it forever and it never shows a seam."""
    rng = np.random.default_rng(seed)
    spec = LIQUIDS[name]
    stops = spec["stops"]
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    shape = (height, width)
    a = fbm(shape, 3, 2, rng, octaves=4)
    b = fbm(shape, 3, 2, rng, octaves=4)
    streams = fbm(shape, 2, 5, rng, octaves=5, gain=0.55)     # stretched along x: liquid flowing sideways
    tint = fbm(shape, 2, 1, rng, octaves=2)
    # domain warping (periodic fields sampled with wrap-around, so the result still tiles)
    qx, qy = sample(a, x, y), sample(b, x, y)
    rx = sample(a, x + 46 * qx + 19, y + 22 * qy + 7)
    ry = sample(b, x + 46 * qx + 3, y + 22 * qy + 31)
    f = sample(streams, x + 64 * rx, y + 26 * ry)
    hue = sample(tint, x, y)

    t = 0.42 * f + 0.3 * hue
    color = palette_at(stops, t)
    other = palette_at(stops, t + 0.45)
    bands = 0.5 + 0.5 * np.sin(f * 3.6 + 1.4 * hue)           # where the colour rises and where it sinks
    lift = bands ** 1.5
    deep = np.array(spec["deep"]) / 255.0
    rgb = deep * (1 - lift)[..., None] + (color * (0.5 + 0.5 * lift)[..., None]) * lift[..., None]
    rgb = rgb * (1 - 0.25 * (1 - lift))[..., None] + other * (0.18 * (1 - lift))[..., None]
    streak = (1 - np.abs(np.tanh(f * 2.2))) ** 12                # thin bright streaks, like light on a moving surface
    rgb = rgb + streak[..., None] * (0.55 * color + 0.35)
    rgb = rgb + blur(rgb, 6) * 0.25                                # a little glow

    # glitter, wrapped round the tile edges so it tiles too
    sparkle = np.zeros(shape)
    for _ in range(70):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        size, amp = rng.uniform(0.45, 1.1), rng.uniform(0.4, 1.1)
        dy = (y - cy + height / 2) % height - height / 2
        dx = (x - cx + width / 2) % width - width / 2
        sparkle += amp * np.exp(-(dx ** 2 + dy ** 2) / (2 * size ** 2))
    rgb = rgb + sparkle[..., None] * (0.7 + 0.3 * color)

    rgb = 1 - np.exp(-rgb * 1.35)
    grey = rgb.mean(axis=2, keepdims=True)
    rgb = np.clip(grey + (rgb - grey) * 1.2, 0, 1)
    return Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), "RGB")


def wisps(name, width=360, height=180, seed=5):
    """A seamless, partly transparent tile of glowing wisps that drifts over the liquid in another direction and
    at another speed. Two layers moving differently keep changing each other's shape, which is what makes the
    surface read as moving liquid instead of one picture sliding past. Its colours are the far side of the
    family's palette, so the wisps look like a second colour mixing in."""
    rng = np.random.default_rng(seed)
    stops = LIQUIDS[name]["stops"]
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    shape = (height, width)
    a = fbm(shape, 2, 2, rng, octaves=4)
    b = fbm(shape, 2, 2, rng, octaves=4)
    curls = fbm(shape, 3, 4, rng, octaves=4, gain=0.5)
    blobs = fbm(shape, 2, 1, rng, octaves=3)
    qx, qy = sample(a, x, y), sample(b, x, y)
    f = sample(curls, x + 90 * qx, y + 44 * qy)
    ridge = (1 - np.abs(np.tanh(f * 1.5))) ** 5                    # long curling ribbons of light
    veil = 1 / (1 + np.exp(-2.2 * sample(blobs, x + 30 * qy, y)))  # soft clouds of colour around them
    alpha = np.clip(0.62 * ridge + 0.26 * veil * (0.35 + ridge), 0, 0.82)
    color = palette_at(stops, 0.5 + 0.28 * f + 0.2 * qx)
    color = color * 0.62 + 0.38                                      # pale, like light caught in the liquid
    rgb = color + ridge[..., None] * 0.25
    # a little glitter caught in the wisps (wrapped so the tile stays seamless)
    sparkle = np.zeros(shape)
    for _ in range(24):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        size, amp = rng.uniform(0.4, 0.8), rng.uniform(0.5, 1.0)
        dy = (y - cy + height / 2) % height - height / 2
        dx = (x - cx + width / 2) % width - width / 2
        sparkle += amp * np.exp(-(dx ** 2 + dy ** 2) / (2 * size ** 2))
    rgb = np.clip(rgb + sparkle[..., None], 0, 1)
    alpha = np.clip(alpha + sparkle * 0.8, 0, 0.9)
    out = np.dstack([rgb, alpha])
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")


# ------------------------------------------------------------------------------------------------ rough cloth
def cloth(tile=512, pitch=8, seed=3):
    """A seamless tile of coarse plain-weave burlap: uneven threads going over and under, fibres and gaps."""
    rng = np.random.default_rng(seed)
    n = tile // pitch
    y, x = np.mgrid[0:tile, 0:tile].astype(np.float64) + 0.5
    col, row = (x // pitch).astype(int) % n, (y // pitch).astype(int) % n
    jitter_x, jitter_y = rng.normal(0, 0.55, n), rng.normal(0, 0.55, n)
    ax = x - ((col + 0.5) * pitch + jitter_x[col])
    ay = y - ((row + 0.5) * pitch + jitter_y[row])
    # thread thickness changes along its length (slubs), and from thread to thread
    slub_warp = periodic_noise((tile, n), 1e-3 + n / 2, 9, rng)      # along y for each vertical thread
    slub_weft = periodic_noise((tile, n), 1e-3 + n / 2, 9, rng)
    base_w = rng.uniform(0.30, 0.42, n)
    half_warp = pitch * np.clip(base_w[col] + 0.06 * slub_warp[y.astype(int) % tile, col], 0.2, 0.5)
    half_weft = pitch * np.clip(base_w[::-1][row] + 0.06 * slub_weft[x.astype(int) % tile, row], 0.2, 0.5)
    prof_warp = np.sqrt(np.clip(1 - (ax / half_warp) ** 2, 0, 1))
    prof_weft = np.sqrt(np.clip(1 - (ay / half_weft) ** 2, 0, 1))
    over = ((col + row) % 2 == 0)                                         # the vertical thread is on top here
    sign = np.where(over, 1.0, -1.0)
    h_warp = prof_warp * (0.62 + 0.38 * sign * np.cos(np.pi * ay / pitch))
    h_weft = prof_weft * (0.62 - 0.38 * sign * np.cos(np.pi * ax / pitch))
    warp_visible = (h_warp >= h_weft) & (prof_warp > 0)
    weft_visible = ~warp_visible & (prof_weft > 0)
    height_map = np.where(warp_visible, h_warp, np.where(weft_visible, h_weft, -0.35))

    fibre_warp = periodic_noise((tile, tile), tile / 2.2, tile / 26, rng)   # streaks running down vertical threads
    fibre_weft = periodic_noise((tile, tile), tile / 26, tile / 2.2, rng)
    fuzz = periodic_noise((tile, tile), tile / 3, tile / 3, rng)
    height_map = height_map + 0.07 * np.where(warp_visible, fibre_warp, fibre_weft) * (height_map > 0)

    gx = (np.roll(height_map, -1, 1) - np.roll(height_map, 1, 1)) / 2
    gy = (np.roll(height_map, -1, 0) - np.roll(height_map, 1, 0)) / 2
    normal = np.dstack([-gx * 2.2, -gy * 2.2, np.ones_like(gx)])
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)
    light = np.array([-0.5, -0.62, 0.61]); light /= np.linalg.norm(light)
    diffuse = np.clip(normal @ light, 0, 1)
    shade = 0.42 + 0.78 * diffuse                                         # low light: the weave is carved by its shading
    occlusion = 0.5 + 0.5 * np.clip(height_map, 0, 1)
    # a soft sheen where the light catches the rounded top of a thread (Blinn half-vector towards a viewer above)
    half = light + np.array([0.0, 0.0, 1.0]); half /= np.linalg.norm(half)
    sheen = np.clip(normal @ half, 0, 1) ** 26 * np.clip(height_map, 0, 1)

    thread_tone = np.where(warp_visible, rng.normal(1, 0.07, n)[col], rng.normal(1, 0.07, n)[row])
    fibre = 1 + 0.14 * np.where(warp_visible, fibre_warp, fibre_weft) + 0.05 * fuzz
    # heathered charcoal: most threads graphite, some carry a little warm brown or a little ink blue
    graphite, warm, ink = np.array([66, 63, 68]) / 255.0, np.array([76, 64, 56]) / 255.0, np.array([52, 58, 76]) / 255.0
    pick_warp, pick_weft = rng.random(n), rng.random(n)
    def thread_colour(pick):
        return np.where((pick < 0.18)[..., None], warm, np.where((pick > 0.86)[..., None], ink, graphite))
    colour = np.where(warp_visible[..., None], thread_colour(pick_warp)[col], thread_colour(pick_weft)[row])
    gap = np.array([9, 8, 11]) / 255.0
    visible = (warp_visible | weft_visible)[..., None]
    rgb = np.where(visible, colour * (thread_tone * fibre * shade * occlusion)[..., None] + (sheen * 0.22)[..., None],
                   gap * (0.85 + 0.3 * fuzz[..., None]))
    # stray fibres lifting off the cloth catch a little light
    hairs = np.clip(periodic_noise((tile, tile), tile / 1.2, tile / 1.2, rng) - 2.6, 0, None)
    rgb = rgb + (hairs * 0.06)[..., None]
    flecks = (rng.random((tile, tile)) < 0.0016)
    rgb = np.where(blur(flecks.astype(float), 1)[..., None] > 0.08, rgb * 0.62, rgb)
    rgb = blur(rgb, 0.6) * 0.35 + rgb * 0.65
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


# ------------------------------------------------------------------------------------------------ crushed velvet
def velvet(tile=768, seed=21):
    """A seamless tile of deep midnight crushed velvet. Velvet's look comes from its pile: where the fibres lie
    towards the light they catch it in a soft sheen, where they lie away they fall dark, and crushing leaves the
    pile lying every which way in soft-edged patches with brighter creases between them. There is no weave."""
    rng = np.random.default_rng(seed)
    shape = (tile, tile)
    y, x = np.mgrid[0:tile, 0:tile].astype(np.float64)
    warp_a, warp_b = fbm(shape, 3, 3, rng, octaves=3), fbm(shape, 3, 3, rng, octaves=3)
    nap = fbm(shape, 3, 3, rng, octaves=3, gain=0.5)           # the direction the pile lies, patch by patch
    fold = fbm(shape, 1.5, 1.5, rng, octaves=2)                # broad, slow undulation of the cloth
    pile = periodic_noise(shape, tile / 2.5, tile / 2.5, rng)  # the fine grain of the pile itself
    fleck = periodic_noise(shape, tile / 6, tile / 1.6, rng)   # faint brushed streaks
    qx, qy = sample(warp_a, x, y), sample(warp_b, x, y)
    angle = np.pi * 1.25 * sample(nap, x + 40 * qx, y + 40 * qy)
    facing = 0.5 + 0.5 * np.cos(angle - 0.6)                     # pile lying towards the light
    sheen = blur(facing ** 2.6, 5)                               # soft-edged patches, as plush pile gives
    crease = blur((1 - np.abs(np.tanh(sample(nap, x + 40 * qx, y + 40 * qy) * 2.0))) ** 3, 4)   # soft, wide creases
    broad = 0.5 + 0.5 * np.tanh(sample(fold, x, y) * 1.2)

    deep = np.array([11, 9, 20]) / 255.0       # midnight, almost black
    mid = np.array([30, 24, 50]) / 255.0       # indigo where the pile turns towards the light
    lit = np.array([86, 70, 122]) / 255.0      # violet sheen at the brightest
    t = np.clip(0.5 * sheen + 0.12 * crease + 0.26 * broad * sheen, 0, 1)
    rgb = deep * (1 - t)[..., None] + mid * t[..., None]
    rgb = rgb + (lit - mid) * (np.clip(t - 0.62, 0, None) / 0.38)[..., None] * 0.85
    rgb = rgb * (1 + 0.07 * pile + 0.025 * fleck)[..., None]       # the plush grain you see up close
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


# ------------------------------------------------------------------------------------------------ output
def data_uri(img, fmt, **options):
    buf = io.BytesIO()
    img.save(buf, fmt, **options)
    mime = {"WEBP": "image/webp", "JPEG": "image/jpeg"}[fmt]
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii"), len(buf.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preview", type=pathlib.Path, help="also save PNG previews into this folder")
    parser.add_argument("--css", type=pathlib.Path, default=CSS)
    parser.add_argument("--canvas", choices=("velvet", "cloth"), default="velvet", help="the page background (default: velvet)")
    args = parser.parse_args()
    lines, sizes = [], {}
    for index, name in enumerate(LIQUIDS):
        img = liquid(name, seed=11 + index * 17)
        uri, size = data_uri(img, "WEBP", quality=82, method=6)
        sizes[name] = size
        lines.append(f"  --fluid-{name}:url(\"{uri}\");")
        overlay = wisps(name, seed=71 + index * 13)
        uri_w, size_w = data_uri(overlay, "WEBP", quality=70, method=6)
        sizes[f"wisp-{name}"] = size_w
        lines.append(f"  --wisp-{name}:url(\"{uri_w}\");")
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=True)
            img.save(args.preview / f"fluid-{name}.png")
            overlay.save(args.preview / f"wisp-{name}.png")
    canvas = velvet() if args.canvas == "velvet" else cloth()
    uri, size = data_uri(canvas, "JPEG", quality=84, optimize=True, progressive=True)
    sizes[f"canvas ({args.canvas})"] = size
    lines.append(f"  --canvas-texture:url(\"{uri}\");")
    if args.preview:
        canvas.save(args.preview / f"canvas-{args.canvas}.png")
    block = ("/* textures:begin (generated by tools/voice_lab_textures.py; do not edit by hand) */\n:root{\n"
             + "\n".join(lines) + "\n}\n/* textures:end */")
    css = args.css.read_text("utf-8")
    pattern = re.compile(r"/\* textures:begin.*?/\* textures:end \*/", re.S)
    css = pattern.sub(lambda _: block, css) if pattern.search(css) else css.rstrip("\n") + "\n\n" + block + "\n"
    args.css.write_text(css, "utf-8", newline="")
    print("bytes:", sizes, "total", sum(sizes.values()))


if __name__ == "__main__":
    main()
