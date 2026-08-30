"""把 WPS 报告副本中的表格统一为标准三线表。

标准三线表：顶线 1.5pt、表头栏目线 0.75pt、底线 1.5pt，无竖线、无内部横线。
现有"三线表"样式只有底线，需补顶线；栏目线按表格逐表加在表头行下边框。

用法: python scripts/maintenance/fix_three_line_tables_20260831.py
输入: D:/Project/厚粲杯/08_算法/docs/交付/0827报告v1_填充版_20260831.docx (原地更新)
依赖: python-docx
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

DOCX_PATH = Path(r"D:/Project/厚粲杯/08_算法/docs/交付/0827报告v1_填充版_20260831.docx")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _set_border(el, tag: str, val: str, sz: str) -> None:
    b = el.find(qn(f"w:{tag}"))
    if b is None:
        b = OxmlElement(f"w:{tag}")
        el.append(b)
    b.set(qn("w:val"), val)
    b.set(qn("w:sz"), sz)
    b.set(qn("w:space"), "0")
    b.set(qn("w:color"), "auto")


def _style_three_line(style_el) -> None:
    """补全样式级 tblBorders：顶线/底线 1.5pt，其余 none。"""
    tblPr = style_el.find(qn("w:tblPr"))
    if tblPr is None:
        return
    borders = tblPr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tblPr.append(borders)
    _set_border(borders, "top", "single", "12")
    _set_border(borders, "bottom", "single", "12")
    for tag in ("left", "right", "insideH", "insideV"):
        _set_border(borders, tag, "none", "0")


def _header_rule(table) -> None:
    """表头行（第 1 行）单元格加 0.75pt 下边框作栏目线。"""
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        borders = tcPr.find(qn("w:tcBorders"))
        if borders is None:
            borders = OxmlElement("w:tcBorders")
            tcPr.append(borders)
        _set_border(borders, "bottom", "single", "6")


def main() -> None:
    doc = Document(str(DOCX_PATH))

    # 1) 样式级补全
    try:
        style = doc.styles["三线表"]
        _style_three_line(style.element)
        print("三线表样式已补全顶线/底线/无竖线")
    except KeyError:
        print("警告：未找到'三线表'样式，跳过样式级修复")

    # 2) 表格级：跳过 4x1 的非数据表（表 0），其余表头行加栏目线
    fixed = 0
    for tbl in doc.tables:
        if len(tbl.columns) == 1:
            continue
        _header_rule(tbl)
        fixed += 1
    print(f"栏目线已加到 {fixed} 张数据表")

    doc.save(str(DOCX_PATH))
    print(f"已保存: {DOCX_PATH}")


if __name__ == "__main__":
    main()
