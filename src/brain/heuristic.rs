//! Rules-based backend: no model, no GPU.
//!
//! It exists so the scan/plan/apply pipeline can be run and tested end to end
//! without a GGUF file, and so a run still produces something sane if the model
//! is unavailable. It reads the same probe text the model does, just with
//! keyword tables instead of comprehension.

use anyhow::Result;
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::{Brain, DigestFields, Filing};
use crate::config::TaxonomyConfig;
use crate::types::{Digest, Probe, Taxonomy};

pub struct Heuristic;

/// (needle, value) pairs, checked in order; first hit wins.
const SYSTEMS: &[(&str, &str)] = &[
    ("pathfinder second edition", "Pathfinder 2e"),
    ("pathfinder 2e", "Pathfinder 2e"),
    ("pathfinder", "Pathfinder 1e"),
    ("pzo", "Pathfinder 1e"),
    ("starfinder", "Starfinder"),
    ("call of cthulhu", "Call of Cthulhu"),
    ("shadowrun", "Shadowrun"),
    ("vampire the masquerade", "World of Darkness"),
    ("fifth edition", "D&D 5e"),
    ("5th edition", "D&D 5e"),
    ("dungeons & dragons", "D&D 5e"),
    ("dungeons and dragons", "D&D 5e"),
    ("d&d 5e", "D&D 5e"),
    ("5e", "D&D 5e"),
    ("3.5", "D&D 3.5e"),
    ("advanced dungeons", "AD&D"),
];

const DOC_TYPES: &[(&str, &str)] = &[
    ("character sheet", "character sheet"),
    ("atelier", "generator"),
    ("generator", "generator"),
    ("random table", "random tables"),
    ("d100", "random tables"),
    ("bestiary", "bestiary"),
    ("menagerie", "bestiary"),
    ("monster manual", "bestiary"),
    ("monsters", "bestiary"),
    ("magic item", "magic items"),
    ("item compendium", "magic items"),
    ("spell", "spells"),
    ("battle map", "maps"),
    ("map pack", "maps"),
    ("player's handbook", "core rules"),
    ("dungeon master's guide", "core rules"),
    ("basic rules", "core rules"),
    ("system reference document", "core rules"),
    ("core rulebook", "core rules"),
    ("campaign setting", "setting"),
    ("gazetteer", "setting"),
    ("adventure path", "adventure"),
    ("adventure", "adventure"),
    ("module", "adventure"),
    ("one-shot", "adventure"),
    ("oneshot", "adventure"),
    ("dungeon", "adventure"),
    ("subclass", "character options"),
    ("subclasses", "character options"),
    ("race", "character options"),
    ("lineage", "character options"),
    ("background", "character options"),
    ("feats", "character options"),
    ("class", "character options"),
    ("sourcebook", "sourcebook"),
    ("supplement", "sourcebook"),
    ("handbook", "sourcebook"),
    ("compendium", "sourcebook"),
    ("guide", "sourcebook"),
];

const SETTINGS: &[&str] = &[
    "Forgotten Realms",
    "Eberron",
    "Ravenloft",
    "Spelljammer",
    "Dragonlance",
    "Greyhawk",
    "Dark Sun",
    "Planescape",
    "Theros",
    "Ravnica",
    "Exandria",
    "Wildemount",
    "Strixhaven",
    "Golarion",
    "Faerun",
    "Faerûn",
];

fn level_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)levels?\s+(\d{1,2})\s*(?:-|–|to|through)\s*(\d{1,2})").unwrap()
    })
}

/// First matching value from a keyword table, searching name before body text.
fn lookup(table: &[(&'static str, &'static str)], name: &str, body: &str) -> Option<&'static str> {
    for (needle, value) in table {
        if name.contains(needle) {
            return Some(value);
        }
    }
    for (needle, value) in table {
        if body.contains(needle) {
            return Some(value);
        }
    }
    None
}

/// Turn a file name into a readable title.
fn title_from_filename(name: &str) -> String {
    let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
    let cleaned: String = stem
        .chars()
        .map(|c| if c == '_' || c == '.' { ' ' } else { c })
        .collect();
    // Only break on hyphens that separate words, not hyphenated words.
    let cleaned = cleaned.replace(" - ", " — ");
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// An embedded title is trusted only if it looks like prose, not a tool artifact.
fn usable_pdf_title(title: &str) -> bool {
    let t = title.trim();
    if t.len() < 4 || t.len() > 120 {
        return false;
    }
    let lower = t.to_lowercase();
    const JUNK: &[&str] = &["untitled", "microsoft word", "document1", ".indd", ".qxd", ".pdf", "layout"];
    if JUNK.iter().any(|j| lower.contains(j)) {
        return false;
    }
    t.chars().any(|c| c.is_alphabetic())
}

impl Brain for Heuristic {
    fn name(&self) -> &str {
        "heuristic"
    }

    fn digest(&mut self, probe: &Probe) -> Result<DigestFields> {
        let name = probe.file_name.to_lowercase();
        let body = probe.text.to_lowercase();
        let haystack = format!("{name} {body}");

        let mut signals = 0;

        let game_system = match lookup(SYSTEMS, &name, &body) {
            Some(s) => {
                signals += 1;
                s.to_string()
            }
            None => "system neutral".to_string(),
        };

        let doc_type = match lookup(DOC_TYPES, &name, &body) {
            Some(t) => {
                signals += 1;
                t.to_string()
            }
            None => "other".to_string(),
        };

        let setting = SETTINGS
            .iter()
            .find(|s| haystack.contains(&s.to_lowercase()))
            .map(|s| {
                signals += 1;
                s.to_string()
            })
            .unwrap_or_else(|| "unknown".to_string());

        let level_range = level_re()
            .captures(&probe.text)
            .map(|c| {
                signals += 1;
                format!("{}-{}", &c[1], &c[2])
            })
            .unwrap_or_else(|| "unknown".to_string());

        let publisher = if haystack.contains("wizards of the coast") {
            "Wizards of the Coast".to_string()
        } else if haystack.contains("paizo") {
            "Paizo".to_string()
        } else if haystack.contains("dungeon masters guild") || haystack.contains("dmsguild") {
            "DMs Guild".to_string()
        } else {
            "unknown".to_string()
        };

        let title = probe
            .pdf_title
            .as_deref()
            .filter(|t| usable_pdf_title(t))
            .map(|t| t.trim().to_string())
            .unwrap_or_else(|| title_from_filename(&probe.file_name));

        // Four independent signals is as sure as rules ever get; cap below the
        // model's ceiling so a real digest always outranks a heuristic one.
        let mut confidence = 0.25 + 0.1 * signals as f32;
        if probe.scanned {
            confidence = (confidence - 0.2).max(0.1);
        }

        Ok(DigestFields {
            title,
            game_system,
            doc_type,
            setting,
            level_range,
            publisher,
            topics: Vec::new(),
            summary: String::new(),
            confidence: confidence.min(0.65),
        })
    }

    fn design_taxonomy(&mut self, corpus: &[Digest], cfg: &TaxonomyConfig) -> Result<Taxonomy> {
        // Count observed system/type pairs, then keep the ones that earn a folder.
        let mut counts: BTreeMap<(String, String), usize> = BTreeMap::new();
        for d in corpus {
            *counts.entry((d.game_system.clone(), d.doc_type.clone())).or_default() += 1;
        }

        let mut leaves: Vec<String> = counts
            .iter()
            .filter(|(_, n)| **n >= cfg.min_docs_per_leaf)
            .map(|((system, doc_type), _)| format!("{system}/{}", title_case(doc_type)))
            .collect();

        // Anything too thin for its own leaf still needs a home per system.
        let systems: BTreeMap<String, usize> =
            counts.iter().fold(BTreeMap::new(), |mut acc, ((s, _), n)| {
                *acc.entry(s.clone()).or_default() += n;
                acc
            });
        for system in systems.keys() {
            let misc = format!("{system}/Other");
            if !leaves.contains(&misc) {
                leaves.push(misc);
            }
        }

        leaves.sort();
        leaves.truncate(cfg.max_leaves);
        Ok(Taxonomy { leaves, notes: Vec::new() })
    }

    fn file(&mut self, digest: &Digest, taxonomy: &Taxonomy) -> Result<Filing> {
        let exact = format!("{}/{}", digest.game_system, title_case(&digest.doc_type));
        if taxonomy.contains(&exact) {
            return Ok(Filing {
                folder: exact,
                confidence: digest.confidence,
                reason: "system and type match a leaf exactly".into(),
            });
        }

        let misc = format!("{}/Other", digest.game_system);
        if taxonomy.contains(&misc) {
            return Ok(Filing {
                folder: misc,
                confidence: (digest.confidence - 0.15).max(0.0),
                reason: "no leaf for this document type; filed under the system".into(),
            });
        }

        // Fall back to whichever leaf shares the most path words with the digest.
        let best = taxonomy
            .leaves
            .iter()
            .max_by_key(|leaf| overlap(leaf, digest))
            .cloned()
            .unwrap_or_else(|| "Other".to_string());

        Ok(Filing {
            folder: best,
            confidence: (digest.confidence - 0.3).max(0.0),
            reason: "closest leaf by keyword overlap".into(),
        })
    }
}

fn overlap(leaf: &str, digest: &Digest) -> usize {
    let hay = format!("{} {} {}", digest.game_system, digest.doc_type, digest.title).to_lowercase();
    leaf.split('/')
        .flat_map(|seg| seg.split_whitespace())
        .filter(|word| word.len() > 3 && hay.contains(&word.to_lowercase()))
        .count()
}

fn title_case(s: &str) -> String {
    s.split_whitespace()
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(name: &str, text: &str) -> Probe {
        Probe {
            file_name: name.into(),
            pdf_title: None,
            pdf_author: None,
            pdf_subject: None,
            pdf_creator: None,
            page_count: Some(10),
            text: text.into(),
            scanned: false,
        }
    }

    #[test]
    fn recognises_a_pathfinder_adventure() {
        let d = (Heuristic).digest(&probe("Skull & Shackles 6.pdf", "A Pathfinder adventure path")).unwrap();
        assert_eq!(d.game_system, "Pathfinder 1e");
        assert_eq!(d.doc_type, "adventure");
    }

    #[test]
    fn recognises_random_tables_by_name() {
        let d = (Heuristic).digest(&probe("100 Nordic Encounters.pdf", "d100 table of encounters")).unwrap();
        assert_eq!(d.doc_type, "random tables");
    }

    #[test]
    fn pulls_a_level_range_out_of_the_text() {
        let d = (Heuristic).digest(&probe("x.pdf", "An adventure for levels 5 to 10")).unwrap();
        assert_eq!(d.level_range, "5-10");
    }

    #[test]
    fn scanned_documents_lose_confidence() {
        let mut p = probe("Ghosts of Saltmarsh.pdf", "");
        p.scanned = true;
        let d = (Heuristic).digest(&p).unwrap();
        assert!(d.confidence < 0.4, "got {}", d.confidence);
    }

    #[test]
    fn rejects_junk_embedded_titles() {
        assert!(!usable_pdf_title("Microsoft Word - draft.doc"));
        assert!(!usable_pdf_title("Untitled"));
        assert!(usable_pdf_title("Ghosts of Saltmarsh"));
    }

    #[test]
    fn every_digest_lands_somewhere() {
        let tax = Taxonomy { leaves: vec!["D&D 5e/Adventure".into(), "D&D 5e/Other".into()], notes: vec![] };
        let digest = Digest {
            hash: "h".into(),
            path: "x.pdf".into(),
            title: "T".into(),
            game_system: "D&D 5e".into(),
            doc_type: "spells".into(),
            setting: "unknown".into(),
            level_range: "unknown".into(),
            publisher: "unknown".into(),
            topics: vec![],
            summary: String::new(),
            confidence: 0.5,
            source: "heuristic".into(),
        };
        let filing = (Heuristic).file(&digest, &tax).unwrap();
        assert!(tax.contains(&filing.folder));
    }
}
