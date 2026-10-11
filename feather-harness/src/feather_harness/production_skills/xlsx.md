## Professional guidance

For new or restyled workbooks without a client template, follow the house style in `.agents/design.md`.

### Rules

- Never replace a formula with its value to make a displayed result look correct.
- Do not save an `.xlsm` through a toolchain that strips VBA, break external links to silence a warning, or unhide or delete support sheets unless asked.
- Describe numbers as recalculated only after an engine actually recalculated them.
- Do not run a workbook's macros or UDFs, including to validate it, unless the user asks or confirms the workbook is trusted.
- In an existing workbook, new rows, columns, and sheets follow the workbook's own fonts, fills, number formats, and color conventions, even where they differ from the house style.

### The workbook as a working model

An existing workbook is a model someone will keep using. Its sheet order, hidden sheets, named ranges, tables, validation, conditional formatting, number formats, calculation settings, date system, external links, pivots, print settings, frozen panes, and outline grouping are all part of how it works, so change them only when the request does.

Use the fewest sheets that keep roles clear. Depending on the task, useful roles are as-of metadata, sourced inputs and assumptions, calculations, the requested output, and focused checks or reconciliation. A one-sheet schedule may be clearer, while a transaction model may need many specialized tabs. Duplicate `Summary`, `Dashboard`, and `Key Takeaways` sheets that restate the same result add nothing. Add a check where a mistake would change a number someone relies on, such as a balance that must tie or a total that must match its source. Don't overdo the checks. A small model needs only a few. Prioritize auditability and working speed: freeze panes, filters, print areas, and named styles help navigation, but dashboard styling across every sheet does not.

Keep sourced inputs distinguishable from formulas and intentional overrides, following the supplied model's conventions first. In a net-new financial model, use the convention finance readers expect: blue font (`#0000FF`) for hard-coded inputs, black for formulas, and green (`#008000`) for links to other sheets, with a source note beside any input drawn from an external source, such as a document or dataset. Make outputs bold or ruled, and show checks as an explicit `OK`/`ERROR` plus value. Beyond the blue, black, and green font colors, the only other color is a dark blue fill with white bold text marking section headers. Each color keeps one meaning on every sheet. Color conventions are not controls, because formulas, validation, sources, and checks carry the logic.

Lay out a net-new time-series model with one period per column and the timeline across the top, and keep each row's formula consistent across periods, with any deliberate exception, such as an opening balance, visibly marked. Reference inputs from their input cells rather than hardcoding them into formulas. Use thousands separators, parentheses for negatives, and a number format that shows zero as a dash, and put units in the row or column header rather than in cells. Center titles across a selection rather than merging cells. On reader-facing sheets, turn gridlines off, freeze panes at the headers, and set the print area. Give every column one job to the extent possible and keep it the same on every sheet. In a time-series model, the columns from left to right are a narrow spacer column, a wide column for labels, a column for units, and then one column per period, all the same width. Leave one blank row between blocks, start each section with a full-width header band, put a thin line above subtotals and totals, and bold only the specific places a reader looks for first. In an existing workbook, keep its own layout, formatting, and sheet settings, such as gridlines, frozen panes, and print areas, and match them in anything you add.

Please do not write prose in the working sheets of a numerical model unless explicitly requested by the user's instruction. They are clean, professional presentations of data, calculations, and dense information. By default, input, calculation, and output sheets carry no sentences: each row has a label and a unit, and a note appears only where a number would be misread without it. In a qualitative workbook, such as a legal mapping, risk register, or diligence tracker, each row carries the concise explanation, evidence, and open question it needs. When the user asks for a readout or cover sheet and neither the request nor the existing workbook sets where it goes, it stands alone on its own sheet as a few short lines, so the working sheets stay clean.

### Formulas and recalculation

An XLSX library reads and writes formula expressions; it does not evaluate them. The file stores a cached value from the last calculation, so a workbook can be structurally valid while showing stale results.

After formula-affecting changes, current values require recalculation. `oracle.recalc`, described in the Paper API companion above, recalculates a temporary copy with LibreOffice. Without an output path it returns calculation and error evidence and writes nothing; with a separate output path it produces a Paper-preserved candidate with eligible calculated caches and never delivers LibreOffice's rewritten package. Its results distinguish recognized formula errors; they do not establish Excel equivalence or financial correctness. After recalculating, confirm that formulas remain, that cached values updated, and that output ranges contain no error values such as `#REF!`, `#DIV/0!`, `#VALUE!`, `#NAME?`, `#N/A`, or unintended blanks, and tie key outputs back to the source and the request.

If recalculation was not possible, say that cached values were not recalculated. Useful identities to tie out include sources = uses, opening + movements = closing, revenue − costs = profit, sum of segments = reported total, bridge components = stated variance, and dates in a schedule = dates used in discounting or accrual logic.

### Common traps

- Setting “full calculation on load” only asks the next opener to calculate, and it does not make the delivered numbers current.
- Rows or columns inserted outside a `SUM` or lookup range silently drop out of totals.
- A formula that differs from its neighbors is often intentional: opening balances, terminal periods, and subtotal rows.
- A constant typed into one cell of a formula run looks like every other cell until an input changes.
- Dates stored as text break sorting, filtering, and date arithmetic while still looking right.
- `####`, unexpected scientific notation, lost decimals, or hidden rows, filters, and outline levels can hide correct numbers from the reader.
- A sensitivity table can vary the wrong input or fail to reset, and a chart can point at the wrong rows or periods.
- Blank, zero, `N/A`, and `NM` mean different things, so collapsing them changes the answer.
- Rounding differences “fixed” by overwriting values break the model the next time an input changes.
- Circular models depend on their iteration limit and tolerance, and data tables, volatile functions, and dynamic-array spills may behave differently in another engine.
- Linked values may be refreshed, stale, or intentionally frozen, and the file does not say which.
- A render of the top of a long output sheet does not review the rest of it.
- A label that restates an input's value, such as "(3% of revenue)", goes stale when the input changes.
- A check that compares against a fixed cell position, such as the grid's base-case cell, fails when a colleague changes the base input.

### Visual review

When layout matters, view the changed and output sheets in a faithful render. For a new workbook or one that will be printed, look at the reader-facing sheets and the print output: page breaks, repeated print titles, scaling, and print areas. Render sheets as the pdf skill describes.

### Examples

A weak operating workbook has `Dashboard`, `Summary`, `Executive Summary`, and `Key Takeaways` sheets that repeat the same totals; inputs and formulas look the same; dates mix text and serials; every used cell has a border; and formula values were not recalculated after the edit. A stronger one has `Inputs` with sourced assumptions and identifiers; a `Model` that preserves formulas and makes overrides explicit; an `Output` with the requested decision table and one meaningful sensitivity; and `Checks` that reconcile totals and flag errors, with cached outputs current after recalculation.

A weak finance-model update writes formulas with a library, opens the workbook successfully, inspects formula strings, and claims results were recalculated and verified. A stronger one preserves formulas and calculation settings, recalculates a candidate with `oracle.recalc`, inspects cached results and error cells, ties the key output back to the source and the request, verifies units, dates, and signs, and describes the recalculation accurately.
