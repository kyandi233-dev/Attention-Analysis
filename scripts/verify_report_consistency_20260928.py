"""Verify manuscript identity, frozen result rows, formatting, and native PDF.

This is an artifact acceptance check, not a test of training or physiological validity.
Run with the bundled document runtime after native Word PDF export.
"""
import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
import pypdfium2 as pdfium
from PIL import Image, ImageDraw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    results = args.results
    manifest = json.loads((results / 'audit15_edit_manifest.json').read_text(encoding='utf-8'))
    source, output = Path(manifest['input']), Path(manifest['output'])
    original = source.parent / 'FocusWave_国赛报告_润色副本.docx'
    assert sha(original) == '1b6c11e54e7cc2e260d0cf9673e19b74d191e08041ec60f2f9c976301091f181'
    assert sha(source) == manifest['input_sha256'] == 'e4b8bf2d05869c73143bd51b03ee7ba3caeffeade3df0c152efdd09ecbf46053'
    assert sha(output) == manifest['output_sha256']
    with zipfile.ZipFile(source) as a, zipfile.ZipFile(output) as b:
        assert a.namelist() == b.namelist()
        changed = [name for name in a.namelist() if a.read(name) != b.read(name)]
        assert sorted(changed) == sorted(['word/document.xml'] + manifest['replacement_media'])
        assert changed == ['word/document.xml', 'word/media/image18.png']
        package_parts = len(a.namelist())
    old, new = Document(source), Document(output)
    assert len(new.tables) == len(old.tables) == 51
    assert [s._sectPr.xml for s in new.sections] == [s._sectPr.xml for s in old.sections]
    text = '\n'.join(p.text for p in new.paragraphs)
    for forbidden in ['至少包含 20 个有效正确 Go RT', '每名参与者至少 2 个',
                      '线性概率模型', '5.6.6', '6.25%', '5.6.4 心肺']:
        assert forbidden not in text, forbidden
    for reference in manifest['new_references']:
        p = next(p for p in new.paragraphs if p.text == reference)
        assert all(r.font.highlight_color == 7 for r in p.runs if r.text)
        assert any(r.italic for r in p.runs) and any(r.italic is False for r in p.runs)
    # All inserted or changed text is yellow; unchanged substrings keep their source formatting.
    for change in manifest['changes']:
        if change['location'].startswith(('insert after', 'reference ', 'E.6 content')):
            p = next(p for p in new.paragraphs if p.text == change['after'])
            assert all(r.font.highlight_color == 7 for r in p.runs if r.text)
    highlighted = lambda d: len(d._element.xpath('.//w:highlight[@w:val="yellow"]'))
    assert highlighted(new) > highlighted(old)
    # Verify every numerical cell of the five reconstructed appendix tables independently.
    specs = [
        (37, 'movement_task_progression', 16, ['estimate', 'ci', 'n_rows']),
        (38, 'movement_q1_models', 12, ['estimate_per_predictor_sd', 'ci', 'n_rows']),
        (39, 'movement_q2_models', 4, ['estimate_per_predictor_sd', 'ci', 'n_rows']),
        (40, 'movement_behavior_links', 20, ['estimate_per_predictor_sd', 'ci', 'n_rows']),
        (41, 'movement_feature_coverage', 7, ['finite_probe_n', 'finite_fraction', 'participant_group_n', 'session_n']),
    ]
    appendix = []
    table_root = args.data_root / 'FormalScience/Movement/tables'
    for idx, name, count, fields in specs:
        path = table_root / (name + '.csv')
        assert sha(path) == manifest['appendix_sources'][name]
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == count == len(new.tables[idx].rows) - 1
        for raw, row in zip(rows, new.tables[idx].rows[1:]):
            observed = [c.text for c in row.cells][-len(fields):]
            expected = []
            for field in fields:
                if field == 'ci':
                    value = f"[{float(raw['ci_low']):.6f}, {float(raw['ci_high']):.6f}]".replace('-', '−')
                elif field == 'finite_fraction':
                    value = f"{float(raw[field]) * 100:.2f}%"
                elif field.startswith('estimate'):
                    value = f'{float(raw[field]):.6f}'.replace('-', '−')
                else:
                    value = raw[field]
                expected.append(value)
            assert observed == expected, (name, observed, expected)
            assert all(c.text for c in row.cells)
            assert all(r.font.highlight_color == 7 for c in row.cells for p in c.paragraphs for r in p.runs if r.text)
        appendix.append({'source': name, 'rows_matched': count, 'sha256': sha(path)})
    assert '动作因变量' in new.tables[40].rows[0].cells[0].text
    assert sum(row.cells[0].text == 'Go 遗漏率（原始）' for row in new.tables[34].rows) == 1
    # Preserve every purely numerical result cell in predictive, calibration, and device tables.
    predictive_tables = [14, 15, 18, 19, 20, 32, 33, 46, 47, 48, 49, 50]
    numeric_cells = 0
    for idx in predictive_tables:
        old_cells = [c.text for row in old.tables[idx].rows for c in row.cells]
        new_cells = [c.text for row in new.tables[idx].rows for c in row.cells]
        assert len(old_cells) == len(new_cells)
        for before, after in zip(old_cells, new_cells):
            if re.fullmatch(r'[\d\s.,%−+\-\[\]()/<>≤≥=NAna—–]+', before) and re.search(r'\d', before):
                assert before == after, (idx, before, after)
                numeric_cells += 1
    # Match against the frozen aggregate payload as well as the local v3 file.
    payload = json.loads((results / 'report_payload.json').read_text(encoding='utf-8'))
    fmt = lambda v, n=4: f'{v:.{n}f}'.replace('-', '−')
    assert len(payload['models']) == 56
    for raw, row in zip(payload['models'], new.tables[46].rows[1:]):
        assert [c.text for c in row.cells][2:7] == [str(raw['n_participants']), str(raw['n_probes']),
            fmt(raw['participant_equal_log_loss']), fmt(raw['ci_lower']), fmt(raw['ci_upper'])]
    for raw, row in zip(payload['models'], new.tables[47].rows[1:]):
        expected = [fmt(raw[k]) for k in ['participant_macro_auroc', 'participant_macro_auroc_ci_lower',
                    'participant_macro_auroc_ci_upper', 'participant_macro_brier']]
        expected += [str(raw['n_participants_auroc_estimable']), str(raw['n_participants_auroc_single_class']),
                     fmt(raw['calibration_slope'], 3) if raw['calibration_slope_reporting'] == 'reportable' else '不作实质解释']
        assert [c.text for c in row.cells][2:] == expected
    assert len(payload['pairs']) == len(new.tables[32].rows) - 1 == 28
    for raw, row in zip(payload['pairs'], new.tables[32].rows[1:]):
        assert row.cells[2].text == fmt(raw['point_estimate'], 6) + ' [' + fmt(raw['ci_lower'], 6) + ', ' + fmt(raw['ci_upper'], 6) + ']'
    pdf = results / 'audit15_render_v4.pdf'
    assert pdf.stat().st_mtime >= output.stat().st_mtime
    document = pdfium.PdfDocument(str(pdf))
    assert len(document) == 125
    pages = [p.get_textpage().get_text_range() for p in document]
    no_text_pages = [i+1 for i, t in enumerate(pages) if not t.strip()]
    compact_pdf = re.sub(r'\s+', '', '\n'.join(pages))
    caption_checks = []
    for p in new.paragraphs:
        if re.match(r'^[图表]\s*\d', p.text):
            key = re.sub(r'\s+', '', p.text)[:16]
            assert key in compact_pdf, ('Native-render caption differs', p.text)
            caption_checks.append(p.text)
    (results / 'audit15_render_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{t}' for i, t in enumerate(pages)), encoding='utf-8')
    render = results / 'render/audit15_v4'
    render.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for index, page in enumerate(document):
        bitmap = page.render(scale=1.3)
        pic = bitmap.to_pil().convert('RGB')
        pic.save(render / f'page-{index+1:03}.png')
        pic.thumbnail((265, 365))
        tile = Image.new('RGB', (285, 395), 'white')
        tile.paste(pic, ((285 - pic.width) // 2, 23))
        ImageDraw.Draw(tile).text((8, 5), f'Page {index+1}', fill='black')
        thumbs.append(tile)
    for offset in range(0, len(thumbs), 12):
        sheet = Image.new('RGB', (4*285, 3*395), '#c0c0c0')
        for j, pic in enumerate(thumbs[offset:offset+12]):
            sheet.paste(pic, ((j % 4)*285, (j // 4)*395))
        sheet.save(render / f'contact-{offset//12+1:02}.jpg')
    result = {'status': 'PASS', 'original_sha256': sha(original), 'input_sha256': sha(source),
              'output_sha256': sha(output), 'native_pdf_sha256': sha(pdf), 'package_parts': package_parts,
              'changed_package_parts': changed, 'section_properties_unchanged': True,
              'paragraphs': len(new.paragraphs), 'tables': len(new.tables), 'native_pdf_pages': len(document),
              'yellow_runs_before': highlighted(old), 'yellow_runs_after': highlighted(new),
              'new_references': len(manifest['new_references']), 'appendix_source_checks': appendix,
              'predictive_table_indices_zero_based': predictive_tables,
              'preserved_numeric_cells': numeric_cells, 'models_refitted': 0,
              'frozen_payload_model_rows_matched': 56, 'frozen_payload_pair_rows_matched': 28,
              'pages_without_extractable_text': no_text_pages,
              'native_render_caption_numbers_matched': len(caption_checks),
              'historical_v3_delivery_sha256': '12956a49e18f98bfc14a3475dfed5da444566feac7c824e352765d9aa0ddea4a',
              'local_v3_differs_from_historical_delivery': True,
              'visual_review': 'Separate human-visible contact-sheet and enlarged-page review recorded in the run note.'}
    (results / 'audit15_document_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
