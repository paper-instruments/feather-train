## Professional guidance

For new or restyled documents without a client template, follow the house style in `.agents/design.md`.

### Rules

- Preserve tracked-changes state. Accepting or rejecting changes needs the user's instruction.
- When editing a draft under negotiation or review, make changes as tracked revisions if the user asks for a redline or the document already uses them. The Paper API companion above covers tracked replacements and native redlines between two versions.
- Use the document's styles, native lists, and numbering, even where they differ from the house style. Do not use empty paragraphs for spacing or type bullet and number characters by hand.
- Do not set summaries, callouts, or quotations in boxes with a one-sided accent border, and add no header or footer text beyond page numbers and markings that the template, the user, or the genre calls for, such as a privilege legend.

### Document semantics and page system

An existing document carries its page setup and section breaks, attached template and named styles, headers, footers, footnotes, fields, captions, cross-references, bookmarks, table of contents, comments, content controls, document properties, custom XML, embedded objects, and protection. Modify styles rather than formatting every paragraph directly when a systemic change is requested. For a local edit, keep the direct formatting that is part of the supplied document.

### Document architecture

Match a new document's structure to its reading mode, and keep an existing document's structure outside the requested change. For example:

- A memo or briefing leads with its answer or recommendation, then gives the reasoning, the material risks or conditions, and the action needed.
- A report gives its answer early, orders its sections logically, and keeps exhibits near the discussion.
- A legal memo or opinion follows its genre's order of question, short answer, analysis, and conclusion.
- Meeting notes record the date, attendees, decisions, and actions with an owner and a date, and leave the discussion itself brief.
- A proposal or statement of work sets out scope, approach, timeline, and terms.
- A procedure numbers its steps and names who does each one.
- A letter is coherent prose with few headings, and a contract follows the supplied form's numbering, defined terms, and cross-references.

Understand what the user is asking for and produce that work product carefully, without overengineering it or adding more than it needs.

In a new document, use built-in heading styles in a logical hierarchy, and simple headered tables without nested, merged, or split structures where possible, because styles make navigation, tables of contents, and controlled reformatting reliable. Do not break a document meant to be read linearly into slide-like cards. Legal documents follow the conventions of their genre and any supplied template: defined terms, numbered clauses, and established citation form take precedence over the house style.

### Building a net-new document

Professional writing is short. A document answering a bounded question gives its answer at the start, and the rest carries only what a reader needs to act on it. A length the user gives is a ceiling to be respected. Write with a word budget in mind: a page of 12-pt single-spaced body text holds about 450 words, fewer once headings, tables, and a memo or title block take their space. Keep type at the house sizes.

Define named styles for the title, headings, body, and block quotations before writing, set 1-inch margins, and apply all formatting through those styles. One typeface covers every style, including the title, headings, headers, and footers. All text is black. Headings are bold, with at most two levels, and the document has no colored text, shaded boxes, or fills.

Reasoning is written as paragraphs. Separate paragraphs with a first-line indent. A bulleted list is for short parallel items, such as conditions or action items, written as plain phrases or sentences. Legal prose may use enumeration inside a sentence, as in (1), (2), and (3), when it is necessary to call particular attention to the elements of the list.

Respect the style and format conventions of the requested work product, or those the user specifies. For example, a memo opens with a To, From, Date, and Re block, set as bold labels with tabbed values between two thin rules, with a privilege or confidentiality legend above it when the genre calls for one, and meeting notes open with the meeting's name, date, and attendees before the decisions and actions. In a multi-page document the footer carries the page number, long citations go in footnotes, exhibits are titled "Exhibit N:" with a message title and a source line beneath, and tables use light horizontal rules and a repeating header row.

Quote supplied text of more than a line as a block indented on both sides, without quotation marks or shading. Proposed language is set the same way under a short label on its own line naming it as a proposal. These blocks keep the body type size and are left-aligned, never justified. When a proposal changes supplied text, show the change as a tracked revision, or with insertions underlined and deletions struck through, unless the user asks for another review format such as comments or clean replacement text.

### Common traps

- Saving through a library that cannot round-trip a feature, such as tracked changes, content controls, embedded objects, or digital signatures, can silently drop it, and any edit invalidates a signature.
- Fields, cross-references, captions, tables of contents, and page numbers do not update when the text changes. Update them with an available engine or say they need updating on open.
- Wholesale normalization destroys intentional exceptions, and paragraph-by-paragraph direct formatting defeats later restyling.
- Headings get stranded at the bottom of a page, captions separate from their figures, and long tables split without repeated header rows unless keep-with-next and repeat-header settings are applied.
- Content placed only in headers or footers is often missed by screen readers.
- Font substitution reflows pages, so pagination in one engine may not hold in another, and a DOCX and its exported PDF can diverge.
- Comments, field codes, or tracked changes can remain visible in the delivered view.
- A document built from an intermediate markup or a patched script can keep stray markers, half-applied edits, or skipped section numbers.
- Shrinking type to fit a page limit reads as crowding and breaks the house sizes, so cut words instead.

### Visual review

When pagination or layout matters and rendering is available, view the affected pages for a scoped edit. For a new or globally changed document, review a page overview, then inspect detail where it matters at a readable resolution. If the DOCX and an exported PDF are both deliverables, check both. When the user gave a length, count the rendered pages before finishing, and if the document runs over, cut repetition and nonessential content rather than shrinking type or margins.

### Examples

A weak finance memo opens with methodology, repeats the recommendation in an executive summary and a conclusion, appends every calculation even though the workbook is already delivered, and leaves the key covenant exception on page four. A stronger one states the recommendation and the covenant condition in its first paragraph, gives the three decision-driving facts in a short “Why” section, reconciles base, downside, and headroom in a compact table, and ends by naming the approval and evidence still required, leaving detailed calculations in the linked workbook.

A weak report turns every section into a card with a blue left border, uses seven shades of blue, repeats the same explanation in an executive summary and a conclusion, appends methodology the user did not request, and treats a successful DOCX parse as visual QA. A stronger one uses styles and whitespace for hierarchy, avoids repeating explanations, keeps evidence and caveats near their claims, views the rendered pages when it can, updates fields, and delivers an editable DOCX plus PDF only when both were requested.
