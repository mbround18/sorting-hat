//! Human-readable rendering of a plan, for the review step.

use std::collections::BTreeMap;

use crate::types::Plan;

/// Draw the proposed library as a tree, with per-folder counts.
pub fn tree(plan: &Plan, full: bool) -> String {
    let mut by_folder: BTreeMap<&str, Vec<&crate::types::Assignment>> = BTreeMap::new();
    for a in &plan.assignments {
        by_folder.entry(a.leaf.as_str()).or_default().push(a);
    }

    let mut out = String::new();
    let mut previous: Vec<&str> = Vec::new();

    for (folder, docs) in &by_folder {
        let segments: Vec<&str> = folder.split('/').collect();
        // Print only the path segments that differ from the previous folder.
        let shared = segments
            .iter()
            .zip(previous.iter())
            .take_while(|(a, b)| a == b)
            .count();

        for (depth, segment) in segments.iter().enumerate().skip(shared) {
            let indent = "  ".repeat(depth);
            let is_leaf = depth == segments.len() - 1;
            if is_leaf {
                out.push_str(&format!("{indent}{segment}/  ({})\n", docs.len()));
            } else {
                out.push_str(&format!("{indent}{segment}/\n"));
            }
        }
        previous = segments;

        if full {
            let indent = "  ".repeat(previous.len());
            for doc in docs {
                let to = doc.dest.file_name().unwrap_or_default().to_string_lossy();
                let from = doc.source.file_name().unwrap_or_default().to_string_lossy();
                // Show the old name only when the document is actually renamed.
                if from == to {
                    out.push_str(&format!("{indent}{to}  [{:.2}]\n", doc.confidence));
                } else {
                    out.push_str(&format!("{indent}{to}  [{:.2}]\n{indent}  was: {from}\n", doc.confidence));
                }
            }
        }
    }

    out
}

/// The one-screen summary printed after planning.
pub fn summary(plan: &Plan) -> String {
    let unsorted = plan
        .assignments
        .iter()
        .filter(|a| a.leaf == crate::pipeline::UNSORTED)
        .count();
    let dupes: usize = plan.duplicates.iter().map(|d| d.others.len()).sum();
    let renamed = plan
        .assignments
        .iter()
        .filter(|a| a.dest.file_name() != a.source.file_name())
        .count();

    let mut out = String::new();
    out.push_str(&format!("  source     {}\n", plan.source_root.display()));
    out.push_str(&format!("  library    {}\n", plan.library_root.display()));
    out.push_str(&format!("  mode       {:?}\n", plan.mode));
    out.push_str(&format!("  folders    {}\n", plan.taxonomy.leaves.len()));
    out.push_str(&format!("  filed      {}\n", plan.assignments.len() - unsorted));
    out.push_str(&format!("  unsorted   {unsorted}\n"));
    out.push_str(&format!("  renamed    {renamed}\n"));
    out.push_str(&format!("  duplicates {dupes} (not filed)\n"));
    out.push_str(&format!("  unreadable {}\n", plan.unfiled.len()));
    out
}
