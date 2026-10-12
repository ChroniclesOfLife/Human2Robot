"""Shared RGB observation overlay for preprocessing and live policy inputs."""

import numpy as np
from PIL import Image, ImageDraw


def project(points, k):
    points = np.asarray(points, dtype=float)
    camera = points @ np.asarray(k).T
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.1)
    pixels = np.full((len(points), 2), np.nan)
    pixels[valid] = camera[valid, :2] / camera[valid, 2, None]
    return pixels, valid


def render_observation(rgb, hands, objects_kpts, k):
    """Render the checkpoint observation convention in RGB, without trails.

    Pose origin is the thumb/index fingertip midpoint. Four segments use root
    y=-0.16 m, bases y=-0.08 m, tips y=0, width=0.18 m open or 0.05 m closed,
    and six dots per segment. Open dots grade red/orange/yellow; closed dots
    are purple/white. At 640x480: hand radius=4, object radius=5, white core=1;
    radii scale by min(width/640, height/480), rounded to at least one pixel.
    Object 1 is cyan, object 2 green, drawn after hands. All world points and
    poses must already be expressed in the optical camera frame.
    """
    image = Image.fromarray(np.asarray(rgb, np.uint8)).copy()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    scale = min(w / 640, h / 480)
    hand_radius, object_radius, core_radius = [max(1, int(round(r * scale))) for r in (4, 5, 1)]

    def dot(pixel, colour, radius):
        if not np.isfinite(pixel).all():
            return
        x, y = int(pixel[0]), int(pixel[1])
        if 0 <= x < w and 0 <= y < h:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)
            draw.ellipse((x - core_radius, y - core_radius, x + core_radius, y + core_radius), fill=(255, 255, 255))

    for side in ("right", "left"):
        if side not in hands:
            continue
        entry = hands[side]
        transform = np.asarray(entry["T_hand_to_world"], dtype=float)
        closed = entry["grasp"] > 0.5
        half_width = (0.05 if closed else 0.18) / 2
        local = np.array([[0, -0.16, 0], [-half_width, -0.08, 0],
                          [half_width, -0.08, 0], [-half_width, 0, 0], [half_width, 0, 0]])
        # RGB palettes ordered fingertip, finger base, palm, wrist.
        palette = ((255, 255, 255), (255, 255, 255), (255, 0, 255), (255, 0, 255)) if closed else (
            (255, 255, 0), (255, 160, 0), (255, 60, 0), (255, 0, 0))
        for segment, (a, b) in enumerate(((0, 1), (0, 2), (1, 3), (2, 4))):
            points = np.linspace(local[a], local[b], 6) @ transform[:3, :3].T + transform[:3, 3]
            pixels, valid = project(points, k)
            start, end = (palette[3], palette[2]) if segment < 2 else (palette[1], palette[0])
            for index, pixel in enumerate(pixels):
                if valid[index]:
                    ratio = index / 5
                    colour = tuple(int(a + (b - a) * ratio) for a, b in zip(start, end))
                    dot(pixel, colour, hand_radius)
    for name, entry in objects_kpts.items():
        points = np.asarray(entry["world"], dtype=float).reshape(-1, 3)
        pixels, valid = project(points, k)
        colour = (0, 255, 255) if name == "obj1" else (0, 255, 0)
        for pixel in pixels[valid]:
            dot(pixel, colour, object_radius)
    return np.asarray(image)
