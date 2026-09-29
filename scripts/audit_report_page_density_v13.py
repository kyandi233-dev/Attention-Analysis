"""Compare per-page usable content height across report layout candidates."""
import argparse
import json
from pathlib import Path

import pdfplumber


def inspect(path):
    out = []
    with pdfplumber.open(path) as doc:
        for index, page in enumerate(doc.pages, 1):
            words = page.extract_words()
            text_bottom = max((word['bottom'] for word in words if word['top'] < 740), default=0)
            image_bottom = max((image['bottom'] for image in page.images), default=0)
            out.append({'page': index, 'text_bottom': round(text_bottom), 'image_bottom': round(image_bottom), 'content_bottom': round(max(text_bottom, image_bottom))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=Path)
    ap.add_argument('candidate', type=Path)
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    before, after = inspect(args.source), inspect(args.candidate)
    result = {
        'before_pages': len(before), 'after_pages': len(after),
        'before_sparse': [x for x in before if x['content_bottom'] < 500],
        'after_sparse': [x for x in after if x['content_bottom'] < 500],
        'before': before, 'after': after,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['before_pages', 'after_pages', 'before_sparse', 'after_sparse']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
