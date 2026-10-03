//! OTSL table markup (`<fcel>`, `<ecel>`, `<lcel>`, `<ucel>`, `<xcel>`,
//! `<nl>`) to HTML. Port of `convert_otsl_to_html` from the model card.

const NL: &str = "<nl>";
const FCEL: &str = "<fcel>";
const ECEL: &str = "<ecel>";
const LCEL: &str = "<lcel>";
const UCEL: &str = "<ucel>";
const XCEL: &str = "<xcel>";
const TOKENS: [&str; 6] = [NL, FCEL, ECEL, LCEL, UCEL, XCEL];

struct Cell {
    text: String,
    row: usize,
    col: usize,
    row_span: usize,
    col_span: usize,
}

/// Splits into OTSL tokens and the non-blank text between them.
fn split(text: &str) -> (Vec<&str>, Vec<&str>) {
    let mut tokens = Vec::new();
    let mut parts = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let next = TOKENS
            .iter()
            .filter_map(|t| rest.find(t).map(|i| (i, *t)))
            .min_by_key(|(i, _)| *i);
        match next {
            Some((i, tok)) => {
                let before = &rest[..i];
                if !before.trim().is_empty() {
                    parts.push(before);
                }
                tokens.push(tok);
                parts.push(tok);
                rest = &rest[i + tok.len()..];
            }
            None => {
                if !rest.trim().is_empty() {
                    parts.push(rest);
                }
                break;
            }
        }
    }
    (tokens, parts)
}

fn count_run(rows: &[Vec<&str>], mut r: usize, mut c: usize, set: &[&str], down: bool) -> usize {
    let mut n = 0;
    while r < rows.len() && c < rows[r].len() && set.contains(&rows[r][c]) {
        n += 1;
        if down {
            r += 1;
        } else {
            c += 1;
        }
    }
    n
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

pub fn to_html(otsl: &str) -> String {
    if otsl.starts_with("<table") && otsl.ends_with("</table>") {
        return otsl.to_string();
    }
    let (tokens, parts) = split(otsl);
    let mut rows: Vec<Vec<&str>> = tokens
        .split(|t| *t == NL)
        .filter(|r| !r.is_empty())
        .map(|r| r.to_vec())
        .collect();
    if rows.is_empty() {
        return String::new();
    }
    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    for r in rows.iter_mut() {
        r.resize(max_cols, ECEL);
    }

    let mut cells = Vec::new();
    let (mut r, mut c) = (0usize, 0usize);
    for (i, part) in parts.iter().enumerate() {
        if *part == FCEL || *part == ECEL {
            let mut text = String::new();
            let mut right = 1;
            if *part != ECEL && i + 1 < parts.len() && !TOKENS.contains(&parts[i + 1]) {
                text = parts[i + 1].trim().to_string();
                right = 2;
            }
            let next_right = parts.get(i + right).copied().unwrap_or("");
            let next_bottom = rows
                .get(r + 1)
                .and_then(|row| row.get(c))
                .copied()
                .unwrap_or("");
            let col_span = 1 + if [LCEL, XCEL].contains(&next_right) {
                count_run(&rows, r, c + 1, &[LCEL, XCEL], false)
            } else {
                0
            };
            let row_span = 1 + if [UCEL, XCEL].contains(&next_bottom) {
                count_run(&rows, r + 1, c, &[UCEL, XCEL], true)
            } else {
                0
            };
            cells.push(Cell {
                text,
                row: r,
                col: c,
                row_span,
                col_span,
            });
        }
        if [FCEL, ECEL, LCEL, UCEL, XCEL].contains(part) {
            c += 1;
        } else if *part == NL {
            r += 1;
            c = 0;
        }
    }
    if cells.is_empty() {
        return String::new();
    }

    let (nr, nc) = (rows.len(), rows[0].len());
    let mut grid: Vec<Vec<Option<usize>>> = vec![vec![None; nc]; nr];
    for (k, cell) in cells.iter().enumerate() {
        for row in grid
            .iter_mut()
            .take((cell.row + cell.row_span).min(nr))
            .skip(cell.row)
        {
            for slot in row
                .iter_mut()
                .take((cell.col + cell.col_span).min(nc))
                .skip(cell.col)
            {
                *slot = Some(k);
            }
        }
    }
    let mut html = String::from("<table>");
    for (ri, row) in grid.iter().enumerate() {
        html.push_str("<tr>");
        for (ci, slot) in row.iter().enumerate() {
            let Some(k) = slot else { continue };
            let cell = &cells[*k];
            if cell.row != ri || cell.col != ci {
                continue;
            }
            html.push_str("<td");
            if cell.row_span > 1 {
                html.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
            }
            if cell.col_span > 1 {
                html.push_str(&format!(" colspan=\"{}\"", cell.col_span));
            }
            html.push('>');
            html.push_str(&escape(&cell.text));
            html.push_str("</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

#[cfg(test)]
mod tests {
    use super::*;

    // covers: REQ-TAB-001
    #[test]
    fn spans() {
        // Header spanning 2 columns, then a cell spanning 2 rows.
        let t = "<fcel>H<lcel><nl><fcel>a<fcel>b<nl><ucel><fcel>c<nl>";
        assert_eq!(
            to_html(t),
            "<table><tr><td colspan=\"2\">H</td></tr><tr><td rowspan=\"2\">a</td><td>b</td></tr><tr><td>c</td></tr></table>"
        );
    }

    // covers: REQ-TAB-001
    #[test]
    fn empty_and_escape() {
        assert_eq!(
            to_html("<fcel>x<y<ecel><nl>"),
            "<table><tr><td>x&lt;y</td><td></td></tr></table>"
        );
        assert_eq!(to_html(""), "");
    }
}
