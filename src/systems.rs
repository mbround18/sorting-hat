//! Canonical game-system names.
//!
//! The model names the system it sees, and the same system arrives spelled
//! several ways: "D&D 5e", "D&D", "Dungeons & Dragons", "Dungeons and Dragons
//! 5th Edition". Left alone, each spelling competes for its own top-level
//! folder and a system with only a document or two — Cyberpunk Red here — loses
//! and ends up filed under something it is not.
//!
//! Names are folded to one spelling before the taxonomy is designed, so the
//! model sees a clean list of the systems that are actually present.

use std::collections::HashMap;

/// Built-in aliases, extended or overridden by `[systems] aliases` in config.
///
/// Bare "D&D" and bare "Pathfinder" are resolved to the edition that dominates
/// a typical collection. That is a judgement call, not a fact: override it in
/// config for a library that is mostly older editions.
const ALIASES: &[(&str, &str)] = &[
    ("d&d", "D&D 5e"),
    ("dnd", "D&D 5e"),
    ("d&d 5e", "D&D 5e"),
    ("d&d 5th edition", "D&D 5e"),
    ("dungeons & dragons", "D&D 5e"),
    ("dungeons and dragons", "D&D 5e"),
    ("dungeons & dragons 5e", "D&D 5e"),
    ("dungeons and dragons 5th edition", "D&D 5e"),
    ("dungeons & dragons 5th edition", "D&D 5e"),
    ("d&d 3.5e", "D&D 3.5e"),
    ("d&d 3.5", "D&D 3.5e"),
    ("d&d 4e", "D&D 4e"),
    ("ad&d", "AD&D"),
    ("pathfinder", "Pathfinder 1e"),
    ("pathfinder 1e", "Pathfinder 1e"),
    ("pathfinder 2e", "Pathfinder 2e"),
    ("pathfinder second edition", "Pathfinder 2e"),
    ("starfinder", "Starfinder"),
    ("cyberpunk red", "Cyberpunk Red"),
    ("cyberpunk 2020", "Cyberpunk 2020"),
    ("call of cthulhu", "Call of Cthulhu"),
    ("shadowrun", "Shadowrun"),
    ("gurps", "GURPS"),
    // Anything the backend could not place shares the generic shelf rather than
    // creating an "unknown" folder nobody would look in.
    ("unknown", "System Neutral"),
    ("system neutral", "System Neutral"),
    ("systems neutral", "System Neutral"),
    ("generic", "System Neutral"),
    ("any", "System Neutral"),
    ("n/a", "System Neutral"),
    ("", "System Neutral"),
];

/// The shelf for documents that name no system.
pub const NEUTRAL: &str = "System Neutral";

/// Fold one system name to its canonical spelling.
///
/// Unrecognised names are kept as the model wrote them, only tidied — a system
/// this table has never heard of is still a real system, and Cyberpunk Red
/// deserves its folder whether or not it was anticipated here.
pub fn canonical(raw: &str, overrides: &HashMap<String, String>) -> String {
    let key = raw.trim().to_lowercase();
    if let Some(name) = overrides.get(&key) {
        return name.clone();
    }
    for (alias, name) in ALIASES {
        if key == *alias {
            return (*name).to_string();
        }
    }
    let tidied = raw.trim();
    if tidied.is_empty() {
        NEUTRAL.to_string()
    } else {
        tidied.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none() -> HashMap<String, String> {
        HashMap::new()
    }

    #[test]
    fn folds_the_many_spellings_of_one_system() {
        for raw in ["D&D 5e", "D&D", "Dungeons & Dragons", "Dungeons and Dragons 5th Edition", "dnd"] {
            assert_eq!(canonical(raw, &none()), "D&D 5e", "{raw}");
        }
        assert_eq!(canonical("Pathfinder", &none()), "Pathfinder 1e");
        assert_eq!(canonical("pathfinder 2e", &none()), "Pathfinder 2e");
    }

    #[test]
    fn keeps_editions_apart() {
        assert_eq!(canonical("D&D 3.5e", &none()), "D&D 3.5e");
        assert_ne!(canonical("D&D 3.5e", &none()), canonical("D&D 5e", &none()));
        assert_ne!(canonical("Pathfinder 2e", &none()), canonical("Pathfinder", &none()));
    }

    #[test]
    fn unnamed_systems_share_the_generic_shelf() {
        assert_eq!(canonical("unknown", &none()), NEUTRAL);
        assert_eq!(canonical("", &none()), NEUTRAL);
        assert_eq!(canonical("  System Neutral ", &none()), NEUTRAL);
    }

    #[test]
    fn an_unanticipated_system_keeps_its_own_name() {
        // The whole point: a system this table never heard of must not be
        // silently swept onto the generic shelf.
        assert_eq!(canonical("Cyberpunk Red", &none()), "Cyberpunk Red");
        assert_eq!(canonical("Mothership", &none()), "Mothership");
        assert_eq!(canonical("Blades in the Dark", &none()), "Blades in the Dark");
    }

    #[test]
    fn config_overrides_win() {
        let mut o = HashMap::new();
        o.insert("d&d".to_string(), "D&D 3.5e".to_string());
        assert_eq!(canonical("D&D", &o), "D&D 3.5e");
    }
}
