# Feather house style

Use this house style for new or restyled deliverables when no client template, brand, or user direction governs. It never applies to an existing file: everything you add to or change in one matches that file's own fonts, sizes, colors, spacing, and table and chart treatment, even where they differ from this guide, unless the user asks for a restyle. The aesthetic is deep blue, neutral, information-dense, and analyst-grade: the requested answer is easy to find, sources and units are close enough to verify, alignment is quiet and exact, and design never competes with the decision.

## Posture

### Quiet confidence

Use a small number of strong decisions repeatedly: one dominant blue, a warm-white or white field, charcoal text, disciplined rules, and one or two semantic accents. A page should feel composed, not decorated.

Avoid the familiar visual shorthand of generic AI-generated work:

- no blue-purple gradients, luminous blobs, glass effects, or decorative constellations;
- no wall of rounded cards where ordinary sections, rows, or columns would be clearer;
- no icon beside every label;
- no repeated three-column feature grid unless the content truly has three parallel categories;
- no ornamental charts, oversized pull quotes, fake dashboards, or infographic furniture;
- no shadows on any element, including cards, markers, and rules;
- no dense field of badges, pills, emojis, or colored labels;
- no heading followed by a one-line fragment merely to populate a layout;
- no card, callout, or panel with a colored accent border on one side only, especially on a rounded box;
- no small all-caps eyebrow label above a title, no oversized “01” and “02” section numerals, and no tagline or subtitle that restates the title;
- no labels in corners, headers, or footers beyond page numbers, source notes, and markings that the template, the user, or the genre calls for, such as a privilege legend or a draft marking;
- no labels or metadata strung together with a middle dot (·), bullet, vertical bar, or slash, such as “Confidential · Draft · Q3 2026” in the corner of a slide;
- no italic or serif word set inside a roman or sans-serif sentence or heading for flourish, as in a sans-serif title with one italic serif word.

A required footer goes in the template's footer, date, and page-number fields.

### Templates come first

A supplied template is the first design system: extend its nearest existing pattern, and do not restyle the whole artifact to match these defaults merely because you can. Related net-new artifacts share one palette, typeface, and set of type roles, and each is laid out for its own reading mode, since a board deck, an audit workbook, and an operating memo are read differently.

### Hierarchy over decoration

Create hierarchy with this order of tools:

1. placement and reading order;
2. whitespace and grouping;
3. type size and weight;
4. rules, fills, or color;
5. illustration or iconography, only when it adds meaning.

If hierarchy depends on a colored box around every paragraph, the structure is not yet resolved.

## Color

Use this palette for net-new work. Use theme colors or styles so later edits remain coherent.

| Role | Hex | Typical use |
|---|---:|---|
| Ultramarine 900 | `#0B1F3A` | Dark backgrounds, section dividers, primary title text |
| Ultramarine 700 | `#173F6B` | Primary accent, slide headings, chart series, rules |
| Ultramarine 500 | `#2C67A0` | Secondary chart series, selected states; not body text on white without a contrast check |
| Mist 100 | `#E8EEF4` | Light section fill, chart reference bands |
| Paper | `#FFFFFF` | Main field |
| Warm paper | `#F7F6F2` | Optional report/deck field where a softer ground is suitable |
| Ink | `#17212B` | Primary body text |
| Slate | `#56616D` | Secondary text, source notes, axis labels |
| Rule | `#CBD3DB` | Hairlines, table rules, separators |
| Positive | `#217A54` | Favorable status or positive variance when semantically appropriate |
| Warning | `#A86713` | Attention, unresolved item, near-threshold condition |
| Negative | `#A33A3A` | Adverse status, breach, negative exception |

Rules:

- Use Ultramarine 900 or Ink for most text. Reserve Ultramarine 700 for hierarchy and the primary data series. Word documents set all text in black, without colored headings.
- Use white text only on a field that passes the intended contrast threshold.
- Keep neutral fills very light, and prefer rules and whitespace to dark grid boxes.
- Do not encode favorable/adverse solely through green/red. Add a sign, arrow, label, symbol, or position.
- Limit an ordinary chart to one primary accent, neutrals, and at most one semantic exception color. A multi-series comparison can use more colors when direct labels and contrast remain adequate.
- Test actual color pairs. Aim for at least 4.5:1 contrast for ordinary text and 3:1 for large text, and make meaningful graphical objects and states distinguishable at 3:1.

## Typography

Prefer a familiar sans serif with reliable Office availability and strong numeric forms. Use Arial for new Office documents. Legal memos, opinions, and agreements use Times New Roman at 12 pt, the norm for legal documents. Use a second face only if the supplied template already establishes one or the user requests a more editorial expression. Never depend on an unembedded specialty font for delivery.

General rules:

- Use no more than three weights in one artifact: regular, semibold, and bold.
- Use sentence case. Reserve all caps for short labels of about one to three words, never for paragraph text.
- Keep underlining for hyperlinks. In running text, use italics only for the titles of works, case names, foreign terms, and a defined term where it is defined, never for emphasis.
- Each text role keeps one face, size, and style throughout the artifact. Do not change face or style within a sentence or heading for decoration, and keep bold for the rare emphasis that has meaning.
- Prefer tabular figures for columns of numbers when the font supports them.
- Never shrink type, or turn on automatic shrink-to-fit, to rescue a structurally overcrowded page. Edit, split, or change the representation.
- Use nonbreaking spaces or keep-with-next behavior where the format supports it to prevent separated currency/unit pairs, orphaned headings, and hanging final words.

**Fixed house styles for net-new artifacts.** These are exact point sizes, and related artifacts use the same styles. Where a subheading shares the body size, bold and spacing distinguish it. PowerPoint has two scales. The read scale suits decks read on screen or in print, such as board pre-reads and investment decks, and is sized for dense analytical slides. The presented scale suits decks shown to a room, such as lectures, lessons, conference talks, and training. Content that does not fit is cut, split, or restructured while the sizes stay as they are. Nothing is set below 10 pt, or below 14 pt on a presented slide.

| Semantic role | PowerPoint read, 16:9 | PowerPoint presented, 16:9 | Word report/memo | Excel workbook |
|---|---:|---:|---:|---:|
| Primary title | 32 pt bold | 40 pt bold | 14 pt bold | 18 pt bold |
| Slide title | 24 pt bold | 32 pt bold | Not used | Not used |
| Key figure in a callout | 28 pt bold | 36 pt bold | Not used | Not used |
| Section heading | 16 pt bold | 24 pt bold | 12 pt bold | 14 pt bold |
| Subheading | 14 pt bold | 20 pt bold | 12 pt bold italic | 11 pt bold |
| Body text | 12 pt regular | 20 pt regular | 12 pt regular | 11 pt regular |
| Table cells and short labels on charts and diagrams | 12 pt regular | 18 pt regular | 12 pt regular | 11 pt regular |
| Source citations, footnotes, and page numbers | 10 pt regular | 14 pt regular | 10 pt regular | 10 pt regular |

Role mapping:

- **PowerPoint:** in a read deck, the slide title is an action title that states each analytical slide's message. On a presented slide, it states the slide's one point or names the question or concept the audience takes up. The primary title is reserved for the cover and section dividers. Section headings divide content within a slide, and subheadings label a chart or smaller content block. Body text, table cells, and short labels on charts and diagrams use their sizes on the deck's scale, and prose keeps that size wherever it sits, including in text columns, boxes, and process steps. Table headers use it in bold. No other display sizes exist, including on the cover.
- **Word:** use named Title, Heading 1, Heading 2, Normal, Caption, and Footnote Text styles. A third heading level is rarely needed, and the document stays at two. Table headers are 12 pt bold.
- **Excel:** the workbook or worksheet title is 18 pt, section bands are 14 pt, and column headers, subtotals, and lower-level labels are 11 pt bold. Ordinary values and formulas remain 11 pt. Format hierarchy must not inflate every row or replace functional number formats.
- **PDF:** retain the source artifact's exact styles on export. A directly authored report PDF uses the Word scale, and a slide PDF uses the scale of its deck. Export must not shrink the page to make content fit.

Apply these styles through slide masters/placeholders, Word styles, and workbook cell styles. Where a library requires explicit font properties, reuse the same named role definitions rather than selecting a size for each text box. Every font size in the artifact is one of these roles.

Do not relabel substantive content as a footnote to make it smaller. Material assumptions, limitations, and chart explanations belong at body or annotation size. Fixed sizes do not guarantee that content fits, so when you can view the rendered output, check fit after font substitution and export.

The only overrides are a supplied client template, explicit user/brand direction, or a stated accessibility/viewing requirement. Define any override at the named-style level for the relevant artifact, not as arbitrary per-object adjustments. Do not scale these point sizes automatically with canvas dimensions or screen pixels. Common requirement-driven overrides:

- If a deck must be readable from across a room, use the presented scale and reflow.
- If a document must keep all text at 12 pt or larger, set its source style to 12 pt as well.
- For wide workbooks, preserve zoom, native editability, and usable print layouts, and never use print scaling to turn a wide model into microscopic text.

## Spacing, alignment, and shape

Use a base spacing unit appropriate to the medium:

- **Slides:** 0.10 in micro unit; most gaps in 0.10, 0.20, 0.30, 0.40, or 0.60 in increments.
- **Word/PDF pages:** 3 pt micro unit; gaps of 3, 6, 9, 12, 18, or 24 pt separate headings, blocks, and sections, while body paragraphs take a first-line indent with no space between them.
- **Excel:** row heights and column widths follow content, with consistent padding simulated through alignment/indent, and use one blank spacer row only when it carries real sectional meaning.

Align to a visible or implied grid. Default to left alignment for text and right alignment for comparable numbers. Decimal-align when precision matters. Centering is suitable for a cover, a short KPI, or a compact column header, but not for paragraphs, source notes, or wide tables.

Use margins generous enough to keep content away from page and slide edges:

- 16:9 slide: start with 0.55–0.70 in left/right and 0.40–0.55 in top/bottom safe margins;
- letter/A4 report or memo: 1 in page margins, unless binding, print, or a supplied template requires others;
- workbook: keep working tables compact, but give title/assumption areas a clear left edge and enough width to avoid accidental wrapping.

Use square corners or a subtle 2–4 pt radius. Prefer 0.5–1 pt rules. Avoid mixed corner radii, heavy outlines, bevels, and shadows. A status marker should be a small symbol or restrained label, not a large colored capsule. Use photography or illustration only when the image itself is evidence, a product or process must be recognized, or a cover genuinely benefits, and never to occupy whitespace.

## Writing

### Write for the decision

Lead a section, slide, or memo with what the reader must understand rather than with the existence of an analysis. “Q3 gross margin fell 240 bps as freight and mix offset price” tells the reader something, while “Gross margin analysis” is merely a label.

Use a three-layer editorial structure where appropriate:

1. **Answer:** the decision, finding, recommendation, or requested output.
2. **Support:** two to four pieces of evidence or a focused visual.
3. **Qualification:** the material assumption, exception, confidence, or action gate.

### Human prose

Write with specificity, natural cadence, and professional restraint.

- Name the actor and action: “Treasury should hold the draw until the covenant model includes the amended add-back,” not “It is recommended that consideration be given to…”
- Prefer concrete nouns and verbs over abstract strings: “Collections slowed” rather than “a deterioration in collections performance was observed.”
- Use fragments only as headings, labels, and table entries.
- Keep caveats next to the affected claim. Do not repeat a general disclaimer on every page.
- Do not add “Key takeaways,” “Summary,” and “Conclusion” sections that restate the same three sentences.

### Lists and bullets

Use bullets for genuinely parallel items: actions, criteria, findings, risks, or components. Use a paragraph for an argument and a table for repeated fields. If each bullet is a multi-sentence mini-essay, use subheadings or a matrix instead.

Keep bullets grammatically parallel. Default to one level. A second level is acceptable when it represents a real parent-child relationship. A third level usually indicates unresolved structure.

## Numbers, units, dates, and sources

Choose conventions once per artifact and apply them consistently.

| Content | Recommended display | Notes |
|---|---|---|
| Currency | `$12.4m`, `$12,400`, or `$12.4 million` | State currency when ambiguity is possible: `US$`, `€`, `GBP` |
| Negative finance value | `($1.2m)` or `−$1.2m` | Follow template; do not mix parentheses and minus signs in one table |
| Percent | `12.4%` | Use the % sign in prose and titles as well as tables, with decimals set by decision relevance |
| Percentage-point change | `+2.4 pp` | Do not call this `+2.4%` |
| Basis-point change | `+240 bps` | Use for rates/margins when audience expects it |
| Multiple | `8.2×` | Use the multiplication sign, not the letter x, when practical |
| Zero | `0`, or the dash an accounting number format displays | Choose one meaning and distinguish zero from not applicable and missing |
| Missing/not applicable | `N/A`, `NM`, or blank with legend | Define; never silently convert missing to zero |
| Large counts | `1.2k`, `1.2m`, `1.2bn` | Put the scale in the header when every row shares it |

Right-align numbers, align decimals where practical, and use comparable precision.

Put units in the title, subtitle, axis, column header, or a nearby note rather than repeatedly in every cell if the whole field shares the same unit. Never combine values at different scales or currencies without a visible conversion rule and effective date.

Use unambiguous dates: `27 September 2026` or `September 27, 2026` in narrative, following audience convention; ISO `2026-09-27` where sorting, interchange, or international ambiguity matters; `Sep 2026` for months; `Q3 2026` or `FY2026 Q3`, consistent with the entity's fiscal calendar; and an en dash for ranges such as `Jan–Sep 2026`. Label actual, forecast, budget, run-rate, LTM, YTD, and as-of dates explicitly. A chart spanning historical and forecast periods should mark the boundary in position, line treatment, or annotation as well as color.

Source notes should be specific enough to retrieve the evidence: document/system name, relevant record or table, reporting/as-of date, and link or identifier where allowed. “Source: company data” is insufficient when several files or cuts exist. Place a short source directly under a chart/table or in its defined source field. Put long citations in footnotes, comments, or a source register and use stable short references in the body. Do not create a bibliography appendix unless the task or number of sources warrants it.

## Tables

Use a table when readers must compare exact values across repeated fields. Use a chart when the shape, direction, magnitude, or relationship is more important than every exact figure. Use prose when there are only one or two facts and no repeated schema.

Table defaults:

- one clear title, optional one-line subtitle, and units near the headers;
- short, concrete headers; one header row whenever possible;
- left-aligned row labels, right-aligned numeric values;
- very light or no vertical rules; horizontal rules at header, subtotal, and total boundaries;
- hierarchy through indentation and weight, not through five fill colors;
- explicit subtotal/total logic, with a double rule reserved for a true final total;
- meaningful sort order: magnitude, chronology, workflow, or decision priority rather than accidental source order;
- notes and definitions immediately below, not hidden in a distant legend;
- no merged cells in the data body unless the supplied format makes them unavoidable.

If the audience needs to copy, filter, trace, or audit the values, preserve the table as native data rather than flattening it into an image.

## Charts and diagrams

Create a chart only when it reveals a pattern or comparison faster than a small table or sentence. Know the one-sentence takeaway the chart should make visible, and if it does not show that, change the chart or remove it.

| Question | Preferred visual |
|---|---|
| Trend over time | Line chart; columns for discrete periods or when zero baseline matters |
| Actual vs budget/plan | Variance columns, dot plot, or paired bars; use a small table when exact reconciliation dominates |
| Ranked categories | Horizontal bars, sorted |
| Composition over time | Stacked columns/area only when both total and mix matter and the series count is small |
| Waterfall from starting to ending value | Waterfall with reconciled bridge |
| Relationship between two measures | Scatter plot with units, relevant labels, and reference line if justified |
| Process, dependency, ownership | Simple flow, swimlane, or RACI-style matrix |
| Status portfolio | Compact table or dot matrix; not a decorative dashboard |

Avoid 3D charts, gauges, donuts with many slices, dual axes unless unavoidable and explicitly labeled, and pie charts for close comparisons. Do not use a Sankey, radar, or bubble chart merely to appear sophisticated.

Chart craft:

- Put the takeaway in the title and the measure/timeframe in a subtitle or axis.
- Start bars at zero. A truncated line-chart axis can be legitimate if clearly labeled and not misleading.
- Direct-label the important series or end points. Remove the legend if direct labels replace it.
- Keep gridlines light and few. Remove chart borders and backgrounds that do not carry meaning.
- Use consistent scales for small multiples intended for comparison.
- Annotate the one or two events needed to explain the pattern, and do not label every point.
- Mark forecast periods with a distinct line style, boundary, or patterned treatment in addition to color.
- Keep source, unit, period, and rounding visible.
- Make the underlying data and calculation path recoverable in editable deliverables.

Use a diagram for relationships, sequence, ownership, or hierarchy, and keep a plain list as text. Prefer straight connectors, a consistent flow direction, no crossing lines where avoidable, and labels that name the action or dependency. If prose can express the relationship in two sentences, a diagram is probably unnecessary.

## Accessibility

Accessibility is part of professional quality, and automated checks are not proof of an accessible experience. Across formats, use sufficient contrast and do not rely on color alone; use meaningful titles, headings, sheet names, slide titles, labels, and link text; add concise alt text for meaningful visuals and mark decorative items; keep reading and focus order logical; use native tables with headers and simple structures; and keep important information out of headers, footers, and images alone. When accessibility is part of the delivery requirement, use an available accessibility checker and inspect the reading experience itself.
