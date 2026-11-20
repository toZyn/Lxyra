from PIL import Image


def letterbox(image, size=448):
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (127, 127, 127))
    canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (pad_x, pad_y))
    return canvas, (nw / w, nh / h, pad_x, pad_y)
