"""Make the featured one-row Kansas FVM GIF from verified existing frames.

This is a crop/title-only operation; it never runs a simulation. Kept FVM
plots, sections and footer pixels are checked byte-for-byte after GIF encoding.
The source must be the known two-row animation, not an already cropped output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageSequence
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "b92e9c3c23491ece576b93ef1dc5223feb336235096273a2bff82569fd69cfe2"
TITLE = "Mississippian interval, Kansas: FVM cold-water injection, 0.05 m³/s"
CAPTION = "The FVM results use the FIM-OBL solver. A MODFLOW-style control-volume finite-difference solver is also available."
KEPT = (0, 602, 960, 1310)
HEADER_HEIGHT = 35


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature(source, output, backup_dir):
    if sha(source) != EXPECTED:
        raise ValueError("Not the verified two-row source; refusing an unverified or repeated crop")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / "methods_compare_kansas-two-row.gif"
    if backup.exists() and sha(backup) != EXPECTED:
        raise ValueError("Backup already exists with different contents")
    shutil.copy2(source, backup)
    original = Image.open(source)
    assert original.size == (960, 1310)
    frames, kept, durations = [], [], []
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 16)
    loop = original.info.get("loop", 0)
    for frame in ImageSequence.Iterator(original):
        old = frame.convert("RGB")
        crop = old.crop(KEPT)
        kept.append(crop)
        durations.append(frame.info.get("duration", 250))
        new = Image.new("RGB", (960, HEADER_HEIGHT + crop.height))
        new.paste(old.crop((0, 0, 960, HEADER_HEIGHT)), (0, 0))
        new.paste(crop, (0, HEADER_HEIGHT))
        # Retain the original timestamp on the right, replace only the title.
        rect = (8, 8, 830, 30)
        draw = ImageDraw.Draw(new)
        draw.rectangle(rect, fill=old.getpixel((8, 8)))
        draw.text((11, 11), TITLE, font=font, fill="black", anchor="lt")
        palette = Image.new("P", (1, 1))
        colors = [rgb for count, rgb in old.getcolors(256)]
        palette.putpalette([v for rgb in colors for v in rgb] + [0] * (768 - 3 * len(colors)))
        patch = new.crop((8, 8, 831, 31)).quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB")
        new.paste(patch, (8, 8))
        indexed = new.quantize(colors=256, dither=Image.Dither.NONE)
        assert ImageChops.difference(new, indexed.convert("RGB")).getbbox() is None, "lossy GIF quantization"
        assert ImageChops.difference(crop, new.crop((0, 35, 960, 743))).getbbox() is None
        frames.append(indexed)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations,
                   loop=loop, disposal=2, optimize=False)
    check = Image.open(output)
    assert check.n_frames == len(kept) and check.size == (960, 743)
    assert check.info.get("loop", 0) == loop
    for i, frame in enumerate(ImageSequence.Iterator(check)):
        assert frame.info.get("duration", 250) == durations[i]
        assert ImageChops.difference(kept[i], frame.convert("RGB").crop((0, 35, 960, 743))).getbbox() is None, "kept plot pixels changed"
    report = dict(path=str(output), sha256=sha(output), frames=check.n_frames,
                  size=list(check.size), durations_ms=durations, title=TITLE, caption=CAPTION,
                  source_sha256=EXPECTED, source_rectangle=list(KEPT), kept_pixels_unchanged=True,
                  simulation_rerun=False)
    (backup_dir / "one-row-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "image/methods_compare_kansas.gif")
    parser.add_argument("--out", type=Path, default=ROOT / "image/methods_compare_kansas.gif")
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    feature(args.source, args.out, args.backup_dir)


if __name__ == "__main__":
    main()
