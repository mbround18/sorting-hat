//! Prompts and GBNF grammars.
//!
//! Every model call is grammar-constrained, so the output is JSON of exactly
//! the expected shape and assignments can only name folders that exist. That
//! removes the whole class of "the model invented a category" failure.

use crate::config::TaxonomyConfig;
use crate::types::{Digest, Probe, Taxonomy};

/// Shared JSON string primitives for every grammar below.
const PRIMITIVES: &str = r#"
ws ::= [ \t\n]*
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
"#;

/// A 0..=1 confidence, expressible as 0, 1, or 0.nnn.
const CONFIDENCE: &str = r#"
conf ::= "1" | "0" ("." [0-9]{1,3})?
"#;

pub fn digest_system() -> String {
    "You are a librarian cataloguing tabletop roleplaying game PDFs. \
You read a fragment of a document and record what it is. \
You answer only with JSON matching the requested schema. \
Use \"unknown\" for any field you cannot determine from the text — never guess. \
Confidence is how sure you are of doc_type and game_system together."
        .to_string()
}

pub fn digest_user(probe: &Probe) -> String {
    let mut s = String::new();
    s.push_str("Catalogue this document.\n\n");
    s.push_str(&format!("File name: {}\n", probe.file_name));
    if let Some(t) = &probe.pdf_title {
        s.push_str(&format!("Embedded title: {t}\n"));
    }
    if let Some(a) = &probe.pdf_author {
        s.push_str(&format!("Embedded author: {a}\n"));
    }
    if let Some(sub) = &probe.pdf_subject {
        s.push_str(&format!("Embedded subject: {sub}\n"));
    }
    if let Some(p) = probe.page_count {
        s.push_str(&format!("Pages: {p}\n"));
    }
    if probe.scanned {
        s.push_str(
            "\nNo text could be extracted; this is an image scan. \
Judge from the file name and metadata alone and set confidence below 0.4.\n",
        );
    } else {
        s.push_str(&format!("\nOpening text:\n---\n{}\n---\n", probe.text));
    }
    s.push_str(
        "\nFields:\n\
- title: the work's real title as printed on its cover or title page. Strip \
file-name noise: scan-site tags, product codes, release-group suffixes, \
underscores, version numbers. If the file name is an opaque code such as \
\"PZO30102E\", take the title from the document text instead. Include a \
subtitle only when it is part of the work's name.\n\
- game_system: e.g. \"D&D 5e\", \"D&D 3.5e\", \"Pathfinder 1e\", \"Pathfinder 2e\", \"Call of Cthulhu\", \"system neutral\"\n\
- doc_type: one of adventure, sourcebook, core rules, setting, bestiary, magic items, \
character options, spells, maps, random tables, generator, zine, character sheet, reference, other\n\
- setting: campaign setting if named, else \"unknown\"\n\
- level_range: e.g. \"1-5\", else \"unknown\"\n\
- publisher: e.g. \"Wizards of the Coast\", \"Paizo\", \"third party\", else \"unknown\"\n\
- topics: three to six short subject keywords\n\
- summary: one sentence on what a game master would use this for\n\
- confidence: 0 to 1\n",
    );
    s
}

pub fn digest_grammar() -> String {
    // Every rule must be on a single line: llama.cpp's GBNF parser treats a
    // newline as the end of a rule, not as whitespace.
    let fields = [
        "title", "game_system", "doc_type", "setting", "level_range", "publisher",
    ]
    .iter()
    .map(|name| format!(r#""\"{name}\":" ws string "," ws"#))
    .collect::<Vec<_>>()
    .join(" ");

    format!(
        r#"root ::= "{{" ws {fields} "\"topics\":" ws "[" ws (string (ws "," ws string){{0,5}})? ws "]" "," ws "\"summary\":" ws string "," ws "\"confidence\":" ws conf ws "}}"
{PRIMITIVES}{CONFIDENCE}"#
    )
}

pub fn vision_digest_system() -> String {
    "You are a librarian cataloguing tabletop roleplaying game PDFs. \
This document has no extractable text, so you are shown a picture of its first \
page instead. Read whatever is printed on it — a cover title, a map's name, a \
product code — and record what the document is. \
You answer only with JSON matching the requested schema. \
Use \"unknown\" for any field the page does not tell you."
        .to_string()
}

pub fn vision_digest_user(probe: &Probe) -> String {
    let mut s = String::from("This is the first page of a PDF.\n\n");
    s.push_str(&format!("File name: {}\n", probe.file_name));
    if let Some(t) = &probe.pdf_title {
        s.push_str(&format!("Embedded title: {t}\n"));
    }
    if let Some(p) = probe.page_count {
        s.push_str(&format!("Pages: {p}\n"));
    }
    s.push_str(
        "\nThe file name may be a meaningless product code — if so, ignore it and \
take the title from the page itself.\n\
\n\
Judge the document type from what you see. A single large illustrated \
location with a grid, or a place drawn from above, is \"maps\" — a battle map \
or poster map, even when a page or two of description comes with it. A cover \
with a title and credits belongs to whatever the book is: an adventure, a \
sourcebook, a bestiary. Pages of stat blocks, spell lists or item entries take \
their type from that content.\n",
    );
    s.push_str(
        "\nFields:\n\
- title: the title printed on the page. If nothing is printed, say \"unknown\"\n\
- game_system: e.g. \"D&D 5e\", \"Pathfinder 1e\", \"system neutral\"\n\
- doc_type: one of adventure, sourcebook, core rules, setting, bestiary, magic items, \
character options, spells, maps, random tables, generator, zine, character sheet, reference, other\n\
- setting: campaign setting if named, else \"unknown\"\n\
- level_range: e.g. \"1-5\", else \"unknown\"\n\
- publisher: the logo or imprint on the page, else \"unknown\"\n\
- topics: three to six short subject keywords for what is depicted\n\
- summary: one sentence on what a game master would use this for\n\
- confidence: 0 to 1, based on how much the page actually told you\n",
    );
    s
}

pub fn taxonomy_system(cfg: &TaxonomyConfig) -> String {
    format!(
        "You are designing the folder tree for a tabletop RPG PDF library. \
You are given a catalogue of every document in the collection. \
Design a tree that fits THIS collection: no empty branches, no categories for \
material that is not present, and no more than {} leaf folders at most {} levels deep.\n\
\n\
Rules:\n\
- The top level is the game system, exactly as it appears in the catalogue \
(\"D&D 5e\", \"Pathfinder 1e\", \"system neutral\"). A game master looks for \
their system first.\n\
- Below that, group by what the document is for at the table: adventures, \
rules, monsters, magic items, character options, tables and generators, maps.\n\
- Add a third level only where one second-level folder would otherwise hold \
dozens of documents — for example splitting adventures by level tier, or a \
large setting's material into its own folder.\n\
- Write folder names in Title Case, as a person would name them: \
\"Magic Items\", not \"magic-items\" or \"magic_items\".\n\
- Every document in the catalogue must have an obvious home.\n\
\n\
Answer only with a JSON array of slash-separated leaf paths, for example:\n\
[\"D&D 5e/Adventures/Tier 1\", \"D&D 5e/Magic Items\", \"Pathfinder 1e/Adventures\"]",
        cfg.max_leaves, cfg.max_depth
    )
}

pub fn taxonomy_user(corpus: &[Digest]) -> String {
    let mut s = String::from("Catalogue:\n");
    for d in corpus {
        s.push_str(&format!(
            "- {} | {} | {} | {}\n",
            d.title, d.game_system, d.doc_type, d.setting
        ));
    }
    s.push_str("\nDesign the leaf folder paths for this collection.\n");
    s
}

pub fn taxonomy_grammar(cfg: &TaxonomyConfig) -> String {
    let max = cfg.max_leaves.max(1) - 1;
    let depth = cfg.max_depth.max(1) - 1;
    format!(
        r#"root ::= "[" ws (leaf (ws "," ws leaf){{0,{max}}})? ws "]"
leaf ::= "\"" seg ("/" seg){{0,{depth}}} "\""
seg ::= [A-Za-z0-9&'()+.,! -]{{1,40}}
{PRIMITIVES}"#
    )
}

pub fn assign_system() -> String {
    "You file one catalogued document into an existing folder tree. \
You may only choose a folder from the list given. \
Pick the most specific folder that genuinely fits. \
Confidence is how well the document matches that folder."
        .to_string()
}

pub fn assign_user(digest: &Digest, taxonomy: &Taxonomy) -> String {
    let mut s = String::from("Folders:\n");
    for leaf in &taxonomy.leaves {
        s.push_str(&format!("- {leaf}\n"));
    }
    s.push_str(&format!(
        "\nDocument:\n- title: {}\n- system: {}\n- type: {}\n- setting: {}\n- levels: {}\n- topics: {}\n- summary: {}\n\nFile it.\n",
        digest.title,
        digest.game_system,
        digest.doc_type,
        digest.setting,
        digest.level_range,
        digest.topics.join(", "),
        digest.summary,
    ));
    s
}

/// Grammar that admits only the folders that actually exist in the taxonomy.
pub fn assign_grammar(taxonomy: &Taxonomy) -> String {
    let alternatives = taxonomy
        .leaves
        .iter()
        .map(|leaf| format!("\"\\\"{}\\\"\"", escape_gbnf(leaf)))
        .collect::<Vec<_>>()
        .join(" | ");
    format!(
        r#"root ::= "{{" ws "\"folder\":" ws folder "," ws "\"confidence\":" ws conf "," ws "\"reason\":" ws string ws "}}"
folder ::= {alternatives}
{PRIMITIVES}{CONFIDENCE}"#
    )
}

fn escape_gbnf(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's GBNF parser ends a rule at the newline. A rule accidentally
    /// wrapped across two lines fails to parse at run time, on the GPU, after a
    /// model load — so catch it here instead.
    fn assert_one_rule_per_line(grammar: &str) {
        for line in grammar.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            assert!(
                line.contains("::="),
                "line is a continuation of the rule above it: {line:?}"
            );
        }
    }

    fn cfg() -> TaxonomyConfig {
        TaxonomyConfig { max_leaves: 40, max_depth: 3, min_docs_per_leaf: 3, min_confidence: 0.45 }
    }

    #[test]
    fn every_grammar_keeps_one_rule_per_line() {
        let taxonomy = Taxonomy { leaves: vec!["A/B".into(), "C".into()], notes: vec![] };
        assert_one_rule_per_line(&digest_grammar());
        assert_one_rule_per_line(&taxonomy_grammar(&cfg()));
        assert_one_rule_per_line(&assign_grammar(&taxonomy));
    }

    #[test]
    fn digest_grammar_names_every_field() {
        let grammar = digest_grammar();
        for field in [
            "title", "game_system", "doc_type", "setting", "level_range", "publisher", "topics",
            "summary", "confidence",
        ] {
            assert!(grammar.contains(field), "{field} missing from the digest grammar");
        }
    }

    #[test]
    fn assign_grammar_admits_only_existing_folders() {
        let taxonomy = Taxonomy { leaves: vec!["D&D 5e/Adventures".into()], notes: vec![] };
        let grammar = assign_grammar(&taxonomy);
        assert!(grammar.contains(r#""\"D&D 5e/Adventures\"""#));
    }

    #[test]
    fn quotes_in_a_folder_name_cannot_break_out_of_the_literal() {
        let taxonomy = Taxonomy { leaves: vec![r#"a" | "b"#.into()], notes: vec![] };
        let grammar = assign_grammar(&taxonomy);
        let folder = grammar.lines().find(|l| l.starts_with("folder ::=")).unwrap();
        assert_eq!(folder.matches('|').count(), 1, "escaped quote leaked an alternative: {folder}");
    }
}

pub fn headings_system() -> String {
    "You are building the table of contents for a tabletop roleplaying game book. \
You are given lines of large type taken from its pages, in reading order, \
numbered. Some are real chapter and section headings. Others are not: sidebar \
titles, spell or monster names in a list, decorative pull quotes, running \
headers, advertisements, credits.\n\
\n\
Choose the lines a reader would want in a table of contents, and give each a \
level: 0 for a part or chapter, 1 for a section within it, 2 for a subsection. \
Keep them in the order given. Prefer too few to too many — a contents page of \
every monster in the bestiary is no use to anyone.\n\
\n\
Answer only with a JSON array of {\"i\": line number, \"l\": level}."
        .to_string()
}

pub fn headings_user(lines: &[(usize, String, usize)]) -> String {
    let mut s = String::from("Lines of large type, in reading order:\n");
    for (index, text, page) in lines {
        s.push_str(&format!("{index}: {text}  (page {page})\n"));
    }
    s.push_str("\nWhich of these belong in the table of contents?\n");
    s
}

/// Grammar for the heading selection: index and level only, so the model
/// cannot invent a heading or a page number — it can only choose among the
/// lines it was shown.
pub fn headings_grammar(max_index: usize, max_level: usize) -> String {
    let digits = max_index.to_string().len().max(1);
    let level = max_level.saturating_sub(1).min(9);
    // Deliberately no whitespace anywhere. Allowing it lets the model
    // pretty-print, and indenting a few hundred entries costs more tokens than
    // the entries themselves — enough to run out of budget mid-array, at which
    // point the reply is unparseable JSON and the whole answer is thrown away.
    format!(
        r#"root ::= "[" (item ("," item)*)? "]"
item ::= "{{\"i\":" index ",\"l\":" [0-{level}] "}}"
index ::= [0-9]{{1,{digits}}}
"#
    )
}

#[cfg(test)]
mod grammar_shape {
    /// A JSON key in a GBNF literal must keep its escaped quotes. Losing them
    /// turns `"\"i\":"` into `""i":"`, which llama.cpp rejects at load time —
    /// invisible until a model call fails on the GPU, so it is pinned here.
    #[test]
    fn json_keys_keep_their_escaped_quotes() {
        let g = super::headings_grammar(283, 3);
        assert!(g.contains(r#"\"i\":"#), "lost escapes in:\n{g}");
        assert!(g.contains(r#"\"l\":"#), "lost escapes in:\n{g}");
        assert!(!g.contains(r#"{"i":"#), "unescaped key in:\n{g}");
    }

    /// Whitespace in this grammar is a token budget leak, not a style choice.
    #[test]
    fn the_reply_grammar_permits_no_pretty_printing() {
        let g = super::headings_grammar(283, 3);
        let root = g.lines().find(|l| l.starts_with("root ::=")).unwrap();
        assert!(!root.contains("ws"), "whitespace allowed in: {root}");
        let item = g.lines().find(|l| l.starts_with("item ::=")).unwrap();
        assert!(!item.contains("ws"), "whitespace allowed in: {item}");
    }

    #[test]
    fn index_width_follows_the_candidate_count() {
        assert!(super::headings_grammar(9, 3).contains("[0-9]{1,1}"));
        assert!(super::headings_grammar(283, 3).contains("[0-9]{1,3}"));
    }

    /// Every grammar the program can emit must survive the same one-rule-per-
    /// line rule as the rest.
    #[test]
    fn headings_grammar_keeps_one_rule_per_line() {
        for line in super::headings_grammar(283, 3).lines() {
            let line = line.trim();
            if !line.is_empty() {
                assert!(line.contains("::="), "continuation line: {line:?}");
            }
        }
    }
}
