## Professional guidance

An exported PDF takes its visual system from its source document. For a directly authored PDF without a client template, follow the house style in `.agents/design.md`.

### Rules

- When an editable source exists, change the source and re-export. Edit a PDF directly only when it is the only source, the change is bounded, and the toolchain preserves the features it needs.
- Do not rasterize a searchable PDF or flatten form fields, comments, layers, links, or signatures unless asked.
- Never call a redaction secure from its appearance: the underlying text must be gone from extraction, copy/paste, layers, attachments, and incremental revisions.

### PDF as a delivery format

PDF work covers export verification, tagging, reading order, bookmarks, links, forms, and final pagination. Export from the authoritative source with embedded fonts or approved substitution, live text and links, bookmarks, document title, language, page labels, and tags where the source supports them. Include bleed or crop marks, security restrictions, and attachment portfolios only when the delivery calls for them. Preserve the original when reversibility matters.

### Common traps

- Any edit can invalidate a digital signature, so identify signed or certified files before changing them.
- OCR is a proposed text layer, not ground truth: names, numbers, dates, tables, and symbols often differ from the page image, especially where confidence is low.
- “Tagged” is not the same as accessible. The tag tree, document language, tab order, and reading order are separate concerns, and multi-column pages often read out of order.
- Compression can blur small text, chart lines, and signatures, and images can fall below a usable resolution for their printed size.
- Export can substitute fonts, drop glyphs, change line breaks, break page references and bookmarks, or push footnotes below the trim area.
- Annotations, comments, hidden layers, and metadata travel with the file to whoever receives it.
- File size alone says little about quality, but a sudden jump or drop often signals rasterized pages or lost images.

### Visual review

When layout matters, view the rendered pages at readable resolution: an overview for continuity, then detail on tables, charts, signatures, footnotes, dense appendices, and pages that structural checks flagged. Render only the pages a check needs, such as `pdftoppm -png -r 150 -f 3 -l 4 report.pdf page`, and read the resulting PNGs. For fine detail, render one page at a higher resolution and crop it to the region, such as `pdftoppm -png -r 300 -f 3 -l 3 -x 300 -y 300 -W 1500 -H 600 report.pdf detail`, where the crop box is in pixels at that resolution. To review a DOCX, PPTX, or XLSX file the same way, convert it to PDF first with `soffice --headless --convert-to pdf deck.pptx`.

### Examples

A weak audit PDF looks polished and passes file-open validation, but its chart is rasterized without alt text, the tag order reads the right column before the left, page 6 has a clipped footnote, and the source date is missing. A stronger one exports from the authoritative report with fonts embedded, live links, bookmarks, tags, and correct page labels. Its pages look right at full size and in detail, its text and reading order extract correctly, and the source date and document status are visible in the body.

A weak export prints to PDF, checks that the file exists, and calls it accessible and visually approved. A stronger one exports from the authoritative source with live text and tags, checks structure and links, views the rendered pages, verifies reading order and key tags when accessibility is required, and describes accurately any check the delivery note mentions.
