from typing import Tuple
import io
import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import cv2
except Exception:
    cv2 = None


def _render_pdf_first_page(pdf_path: str, zoom: float = 2.0) -> Image.Image:
    """Render the first page of a PDF to a PIL Image using PyMuPDF.

    Falls back to raising an informative error if PyMuPDF is not available.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required: pip install pymupdf")
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def _to_gray_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def verify_ecg_presence(pil_img: Image.Image):
    """Return simple heuristic scores and combined confidence that an ECG strip is present.

    Returns a dict with keys: proj_score, edge_score, long_score, grid_score, ecg_confidence
    """
    import numpy as _np
    import cv2 as _cv2

    img = pil_img.convert("RGB")
    arr = _np.array(img)
    gray = _cv2.cvtColor(arr, _cv2.COLOR_RGB2GRAY)

    # 1) projection score (vertical projection of waveform energy)
    inv = 255 - gray
    proj = inv.mean(axis=1)
    proj_score = float(_np.clip((proj.max() - proj.mean()) / (proj.max() + 1e-6), 0.0, 1.0))

    # 2) edge density
    edges = _cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (edges.size * 255.0)
    edge_score = float(_np.clip(edge_density * 3.0, 0.0, 1.0))

    # 3) long thin contour fraction (waveform continuity)
    contours, _ = _cv2.findContours(edges, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    long_count = 0
    for c in contours:
        x, y, ww, hh = _cv2.boundingRect(c)
        if ww >= 0.5 * w and hh <= 0.35 * h:
            long_count += 1
    long_frac = min(1.0, long_count / 2.0)
    long_score = float(long_frac)

    # 4) grid detection (Hough lines) -- optional indicator
    try:
        lines = _cv2.HoughLinesP(edges, 1, _np.pi / 180, threshold=max(50, w // 30), minLineLength=w // 8, maxLineGap=10)
    except Exception:
        lines = None
    grid_score = 0.0
    if lines is not None:
        horiz = sum(1 for x1, y1, x2, y2 in lines[:, 0] if abs(y2 - y1) < 5)
        vert = sum(1 for x1, y1, x2, y2 in lines[:, 0] if abs(x2 - x1) < 5)
        grid_score = float(_np.clip((horiz + vert) / 20.0, 0.0, 1.0))

    # combine with tuned weights
    combined = 0.35 * proj_score + 0.35 * edge_score + 0.2 * long_score + 0.1 * grid_score

    metadata = {
        "proj_score": proj_score,
        "edge_score": edge_score,
        "long_score": long_score,
        "grid_score": grid_score,
        "ecg_confidence": float(_np.clip(combined, 0.0, 1.0)),
    }
    return metadata


def extract_ecg_strip_from_pdf(
    pdf_path: str,
    line_count: int = 3,
    threshold_factor: float = 2.0,
    verify: bool = False,
    min_confidence: float = 0.5,
) -> Image.Image:
    """Extract a single, horizontally stitched ECG strip image from a PDF.

    The PDF is expected to contain a multi-line ECG (e.g., 3 stacked lines covering
    a 30s recording). This function renders the first page, detects the three
    waveform strips by finding horizontal bands of signal, crops each band, and
    stitches them left-to-right into one long image.

    If `verify` is True, the function will also run `verify_ecg_presence` on the
    stitched output and return a 3-tuple `(image, image_path, metadata)` where
    `metadata` contains an `ecg_confidence` score in 0..1 and intermediate measures.

    Default behavior (verify=False) keeps the original return signature
    `(image, image_path)` for backwards compatibility.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required: pip install opencv-python")

    img = _render_pdf_first_page(pdf_path)
    gray = _to_gray_cv(img)

    # Compute horizontal projection to find dense rows (signal areas)
    # Invert so signal lines (dark traces) become bright for projection
    inv = 255 - gray
    proj = inv.mean(axis=1)

    # Smooth projection to reduce noise
    proj_smooth = cv2.GaussianBlur(proj.reshape(-1, 1), (15, 1), 0).ravel()

    # Threshold to find regions with significant signal
    thr = (proj_smooth.max() + proj_smooth.mean()) / threshold_factor
    mask = proj_smooth > thr

    # Find contiguous True regions -> candidate strips
    strips = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            strips.append((start, i))
            start = None
    if start is not None:
        strips.append((start, len(mask)))
    # print(f"found {len(strips)} strips")
    # If detection found more than line_count, merge/split conservatively
    if len(strips) < line_count:
        # fallback: split image vertically into `line_count` equal bands
        h = gray.shape[0]
        strips = []
        band_h = h // line_count
        for i in range(line_count):
            a = i * band_h
            b = h if i == line_count - 1 else (i + 1) * band_h
            strips.append((a, b))

    # If more than needed, take the largest `line_count` regions
    if len(strips) > line_count:
        strips = sorted(strips, key=lambda r: r[1] - r[0], reverse=True)[:line_count]
        strips = sorted(strips, key=lambda r: r[0])

    # Crop each strip with larger vertical padding (prevent narrow strips) and full width
    crops = []
    h, w = gray.shape
    for (a, b) in strips:
        # pad relative to the detected strip height (20% of strip height, min 10px)
        pad = max(10, int(20 * (b - a)))
        a2 = max(0, a - pad)
        b2 = min(h, b + pad)
        cropped = img.crop((0, a2, w, b2))
        # Optionally, further crop left/right to the waveform region by
        # computing vertical projection and trimming whitespace
        cgray = _to_gray_cv(cropped)
        col_proj = (255 - cgray).mean(axis=0)
        col_thr = max(5, (col_proj.max() * 0.05))
        cols = np.where(col_proj > col_thr)[0]
        if cols.size:
            x0, x1 = int(cols[0]), int(cols[-1])
            cropped = cropped.crop((x0, 0, x1 + 1, cropped.size[1]))
        crops.append(cropped.convert("RGB"))

    # Resize crops to same height for horizontal stitching (use max height)
    # so we don't shrink taller strips and produce a narrow output
    max_h = max(c.size[1] for c in crops)
    resized = [c.resize((int(c.size[0] * (max_h / c.size[1])), max_h), Image.LANCZOS) for c in crops]

    # Stitch left-to-right
    total_w = sum(c.size[0] for c in resized)
    out = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for c in resized:
        out.paste(c, (x, 0))
        x += c.size[0]

    out.save("ecg_output.png")

    if verify:
        metadata = verify_ecg_presence(out)
        if metadata.get("ecg_confidence", 0.0) < float(min_confidence):
            return None, None, metadata
        return out, "ecg_output.png", metadata

    return out, "ecg_output.png"


def extract_text_from_pdf(pdf_path: str, pages=None) -> str:
    """Extract text from a PDF using PyMuPDF.

    Args:
        pdf_path: path to the PDF file.
        pages: optional iterable of 0-based page indices to extract. If None, extract all pages.

    Returns:
        A single string containing the concatenated page texts.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required: pip install pymupdf")
    doc = fitz.open(pdf_path)
    texts = []
    if pages is None:
        page_iter = range(doc.page_count)
    else:
        page_iter = pages
    for p in page_iter:
        page = doc.load_page(p)
        texts.append(page.get_text("text"))
    return "\n".join(texts)

    


if __name__ == "__main__":
    # Quick manual test: render and save stitched output for a sample PDF
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        # if user requests text extraction
        if len(sys.argv) > 2 and sys.argv[2] in ("--text", "-t"):
            txt = extract_text_from_pdf(path)
            outname = path.rsplit(".", 1)[0] + ".txt"
            with open(outname, "w", encoding="utf-8") as f:
                f.write(txt)
            print(f"Saved {outname}")
        else:
            out, _ = extract_ecg_strip_from_pdf(path)
            out.save("ecg_stitched.png")
            print("Saved ecg_stitched.png")
