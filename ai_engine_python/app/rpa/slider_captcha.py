"""Bounded slider handling on an existing Playwright page; no browser ownership."""

import asyncio
import base64
import contextlib
import random


async def find_slider_challenge(page, allowed_frame=None):
    for frame in page.frames:
        if allowed_frame and not allowed_frame(frame.url):
            continue
        try:
            if await frame.locator("#sliderDiv:visible").count() == 1:
                return frame
        except Exception:
            continue  # Redirects can detach a frame during the probe.
    return None


def captcha_distance(background, slider):
    # Import on use so a missing optional image runtime does not stop the Worker.
    import cv2
    import numpy as np

    bg = cv2.imdecode(np.frombuffer(background, np.uint8), cv2.IMREAD_GRAYSCALE)
    block = cv2.imdecode(np.frombuffer(slider, np.uint8), cv2.IMREAD_UNCHANGED)
    if bg is None or block is None:
        raise ValueError("Invalid captcha image")
    crop_x = 0
    if block.ndim == 3 and block.shape[2] == 4:
        _, mask = cv2.threshold(block[:, :, 3], 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Empty captcha block")
        crop_x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        block = cv2.cvtColor(block[y : y + height, crop_x : crop_x + width], cv2.COLOR_BGRA2GRAY)
    elif block.ndim == 3:
        block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
    if block.shape[0] > bg.shape[0] or block.shape[1] > bg.shape[1]:
        raise ValueError("Captcha image sizes do not match")
    edges = cv2.Canny(block, 100, 200)
    if not np.any(edges):
        raise ValueError("Captcha block has no edges")
    matches = cv2.matchTemplate(cv2.Canny(bg, 100, 200), edges, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, position = cv2.minMaxLoc(matches)
    distance = position[0] - crop_x
    if confidence < 0.2 or distance <= 0:
        raise ValueError("Captcha match is uncertain")
    return distance


def generate_fast_tracks(distance):
    # Keep fractional increments: rounding each step can accumulate an offset.
    positions = [distance * i / 10 for i in range(11)]
    return [b - a for a, b in zip(positions, positions[1:])]


async def solve_slider_captcha(
    page, *, frame=None, max_attempts=3, can_continue=None, on_attempt=None
):
    """Return solved/not_present/failed/interrupted; never serialize page content."""
    attempts = 0
    active = can_continue or (lambda: True)
    frame = frame or await find_slider_challenge(page)
    if frame is None:
        return {"status": "not_present", "attempts": 0}
    try:
        container = frame.locator("#sliderDiv")
        for _ in range(max(0, min(3, max_attempts))):
            if not active():
                return {"status": "interrupted", "attempts": attempts}
            if not await container.is_visible():
                return {"status": "solved", "attempts": attempts}
            attempts += 1
            if on_attempt:
                on_attempt()
            try:
                # Only read the two challenge canvases, never credentials or page HTML.
                images = await container.evaluate("""root => {
                    const bg = root.querySelector('canvas');
                    const block = root.querySelector('canvas.block');
                    if (!bg || !block || bg === block) throw Error('Unsupported slider');
                    return {bg: bg.toDataURL('image/png'), block: block.toDataURL('image/png'),
                            scale: bg.getBoundingClientRect().width / bg.width};
                }""")
                distance = (
                    captcha_distance(
                        base64.b64decode(images["bg"].split(",", 1)[1]),
                        base64.b64decode(images["block"].split(",", 1)[1]),
                    )
                    * images["scale"]
                )
                box = await container.locator(".slider").bounding_box(timeout=2000)
                bounds = await container.bounding_box(timeout=2000)
                if not box or not bounds or not 0 < distance <= bounds["width"]:
                    raise ValueError("Invalid captcha geometry")
                x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                if not active():
                    return {"status": "interrupted", "attempts": attempts}
                await page.mouse.move(x, y)
                try:
                    await page.mouse.down()
                    for delta in generate_fast_tracks(distance):
                        if not active():
                            return {"status": "interrupted", "attempts": attempts}
                        x += delta
                        await page.mouse.move(x, y + random.uniform(-1, 1))
                        await asyncio.sleep(0.01)
                finally:
                    with contextlib.suppress(Exception):
                        await page.mouse.up()
                # Wait for the challenge itself, not networkidle on a polling login page.
                await container.wait_for(state="hidden", timeout=3000)
                return {"status": "solved", "attempts": attempts}
            except (ImportError, ModuleNotFoundError):
                return {
                    "status": "failed",
                    "attempts": attempts,
                    "message": "滑块图像处理依赖未安装，请手动完成验证",
                }
            except Exception:
                if not active():
                    return {"status": "interrupted", "attempts": attempts}
                await asyncio.sleep(0.2)
    except Exception:
        pass  # Detached/unsupported frames are failures, never evidence of success.
    return {"status": "failed", "attempts": attempts, "message": "滑块验证未通过，请手动完成验证"}
