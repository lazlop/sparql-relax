use oxigraph::io::{RdfFormat, RdfParser};
use oxigraph::store::Store;
use sparql_relax_core::bfs::Hop;
use sparql_relax_core::{NamespaceScope, RelaxError, diagnose, diagnose_and_relax, diagnose_and_relax_default, diagnose_default};
use std::time::{Duration, Instant};

const TTL: &str = r#"
    @prefix ex: <urn:example#> .
    ex:building223 ex:hasPart ex:zone1 .
    ex:zone1 ex:hasSensor ex:sensor1 .
    ex:sensor1 a ex:TempSensor .
"#;

fn test_store() -> Store {
    let store = Store::new().unwrap();
    store.load_from_slice(RdfParser::from_format(RdfFormat::Turtle), TTL).unwrap();
    store
}

const BROKEN_QUERY: &str = r#"
    PREFIX ex: <urn:example#>
    SELECT ?sensor WHERE {
        ex:building223 ex:hasSensor ?sensor .
        ?sensor a ex:TempSensor .
    }
"#;

#[test]
fn diagnoses_the_wrong_predicate_as_a_culprit() {
    let store = test_store();
    let diagnosis = diagnose(BROKEN_QUERY, &store, 1, None).unwrap();

    assert_eq!(diagnosis.original_row_count, 0);
    assert_eq!(diagnosis.culprits.len(), 1);
    let culprit = &diagnosis.culprits[0];
    assert_eq!(culprit.depth, 1);
    assert_eq!(culprit.triples.len(), 1);
    assert_eq!(culprit.triples[0].predicate.to_string(), "<urn:example#hasSensor>");
}

#[test]
fn diagnose_default_still_finds_the_culprit_with_its_internal_timeout() {
    let store = test_store();
    let diagnosis = diagnose_default(BROKEN_QUERY, &store).unwrap();
    assert_eq!(diagnosis.culprits.len(), 1);
}

#[test]
fn diagnose_errors_with_timeout_when_even_the_original_query_cant_run_in_time() {
    // An effectively-zero timeout means the deadline is already passed by
    // the time diagnose_parsed's very first query (the original query
    // itself) would run — there's nothing meaningful to diagnose without
    // knowing its row count, so this is a hard error, not a graceful
    // "no culprits found".
    let store = test_store();
    let result = diagnose(BROKEN_QUERY, &store, 1, Some(Duration::from_nanos(1)));
    assert!(matches!(result, Err(RelaxError::Timeout)), "expected a Timeout error");
}

#[test]
fn find_path_respects_a_past_deadline_even_when_a_real_path_exists() {
    // Isolates the BFS-specific deadline check (as opposed to the SPARQL
    // query timeouts diagnose_and_relax's own tests exercise): the *only*
    // thing that differs between these two calls is `deadline`, so a
    // real, existing path being missed proves the search itself is what's
    // being cut off, not some unrelated query.
    let store = test_store();
    let start = oxigraph::model::Term::NamedNode(oxigraph::model::NamedNode::new("urn:example#building223").unwrap());
    let goal = oxigraph::model::Term::NamedNode(oxigraph::model::NamedNode::new("urn:example#sensor1").unwrap());

    let found = sparql_relax_core::bfs::find_path(&store, &start, &goal, 2, None, None);
    assert!(found.is_some(), "sanity check: the real hasPart/hasSensor path should be found with no deadline");

    let past_deadline = Instant::now() - Duration::from_secs(1);
    let cut_off = sparql_relax_core::bfs::find_path(&store, &start, &goal, 2, None, Some(past_deadline));
    assert!(cut_off.is_none(), "a deadline already in the past should stop the search before it finds the path");
}

#[test]
fn find_path_still_works_through_a_high_fan_out_hub_node() {
    // Regression test for switching `neighbors` from eagerly collecting
    // every edge into a `Vec` to a lazy iterator (so a deadline can be
    // checked partway through a huge fan-out node's edges instead of only
    // after all of them are materialized): the real edge should still be
    // found among hundreds of unrelated ones, in whatever order the store
    // happens to yield them.
    let store = Store::new().unwrap();
    let mut ttl = String::from("@prefix ex: <urn:example#> .\n");
    for i in 0..500 {
        ttl.push_str(&format!("ex:hub ex:hasNoise ex:noise{i} .\n"));
    }
    ttl.push_str("ex:hub ex:hasPart ex:target .\n");
    store.load_from_slice(RdfParser::from_format(RdfFormat::Turtle), &ttl).unwrap();

    let start = oxigraph::model::Term::NamedNode(oxigraph::model::NamedNode::new("urn:example#hub").unwrap());
    let goal = oxigraph::model::Term::NamedNode(oxigraph::model::NamedNode::new("urn:example#target").unwrap());

    let found = sparql_relax_core::bfs::find_path(&store, &start, &goal, 1, None, None);
    assert_eq!(found, Some(vec![Hop::Forward(oxigraph::model::NamedNode::new("urn:example#hasPart").unwrap())]));
}

#[test]
fn diagnose_and_relax_propagates_a_timeout_error_when_diagnose_timeout_is_exceeded() {
    // diagnose_timeout is independent of the relax-phase `timeout` param —
    // an effectively-zero diagnose_timeout should fail the whole call the
    // same way a standalone `diagnose` call would, regardless of what the
    // relax-phase timeout is set to.
    let store = test_store();
    let result = diagnose_and_relax(
        BROKEN_QUERY,
        &store,
        1,
        Some(4),
        Some(5),
        None,
        NamespaceScope::Unrestricted,
        None,
        Some(Duration::from_nanos(1)),
    );
    assert!(matches!(result, Err(RelaxError::Timeout)), "expected a Timeout error");
}

#[test]
fn finds_a_real_forward_forward_path_and_fixes_the_query() {
    let store = test_store();
    let report = diagnose_and_relax(BROKEN_QUERY, &store, 1, Some(4), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();

    assert_eq!(report.original_row_count, 0);
    assert_eq!(report.results.len(), 1);

    let result = &report.results[0];
    assert_eq!(result.triples.len(), 1);
    let relaxed_triple = &result.triples[0];
    assert_eq!(relaxed_triple.hop_alternatives.len(), 1);
    assert_eq!(relaxed_triple.hop_alternatives[0].len(), 2);
    assert_eq!(relaxed_triple.path_text.as_deref(), Some("(<urn:example#hasPart> / <urn:example#hasSensor>)"));
    assert_eq!(result.row_count, 1);

    let relaxed = result.relaxed_query.as_ref().unwrap();
    assert!(relaxed.contains("hasPart"));
}

#[test]
fn relaxes_a_culprit_triple_whose_variable_is_not_in_the_select_list() {
    // `?sensor` is a plain WHERE-clause bridge variable — it's never listed
    // in `SELECT ?reading` — but it's still what the broken `hasSensor`
    // triple's endpoint search needs to resolve. Regression test for a bug
    // where the reduced query's rows were built from the *original* query's
    // `Project`, which strips any variable not in its list (per SPARQL
    // semantics), silently making endpoint resolution (and the ablation
    // per-row check) blind to any culprit triple touching a non-selected
    // variable — always falling back to the pruned/dropped-constraint query
    // instead of finding the real `hasPart/hasSensor` path.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:building223 ex:hasPart ex:zone1 .
                ex:zone1 ex:hasSensor ex:sensor1 .
                ex:sensor1 ex:reports ex:reading1 .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?reading WHERE {
            ex:building223 ex:hasSensor ?sensor .
            ?sensor ex:reports ?reading .
        }
    "#;

    let report = diagnose_and_relax(query, &store, 1, Some(4), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(report.original_row_count, 0);
    assert_eq!(report.results.len(), 1);

    let result = &report.results[0];
    let relaxed_triple = &result.triples[0];
    assert!(!relaxed_triple.hop_alternatives.is_empty(), "the hasPart/hasSensor path should be found even though ?sensor isn't selected");
    assert_eq!(relaxed_triple.path_text.as_deref(), Some("(<urn:example#hasPart> / <urn:example#hasSensor>)"));
    assert!(result.relaxed_query.is_some());
    assert_eq!(result.row_count, 1);
}

#[test]
fn query_with_no_broken_triples_reports_no_culprits() {
    let store = test_store();
    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:zone1 ex:hasSensor ?sensor .
            ?sensor a ex:TempSensor .
        }
    "#;
    let diagnosis = diagnose(query, &store, 1, None).unwrap();
    assert_eq!(diagnosis.original_row_count, 1);
    assert!(diagnosis.culprits.is_empty());
}

#[test]
fn finds_an_inverse_hop_path() {
    // sensor1 --partOf--> zone1: to relax "zone1 hasSensor ?sensor" via the
    // *inverse* of partOf, the search has to try incoming edges too.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:sensor1 ex:partOf ex:zone1 .
                ex:sensor1 a ex:TempSensor .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:zone1 ex:hasSensor ?sensor .
            ?sensor a ex:TempSensor .
        }
    "#;

    let report = diagnose_and_relax(query, &store, 1, Some(4), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(report.results.len(), 1);
    let relaxed_triple = &report.results[0].triples[0];
    assert_eq!(
        relaxed_triple.hop_alternatives,
        vec![vec![Hop::Inverse(oxigraph::model::NamedNode::new("urn:example#partOf").unwrap())]]
    );
    assert_eq!(relaxed_triple.path_text.as_deref(), Some("^(<urn:example#partOf>)"));
    assert_eq!(report.results[0].row_count, 1);
}

#[test]
fn reports_no_relaxation_when_depth_is_too_small() {
    let store = test_store();
    let report = diagnose_and_relax(BROKEN_QUERY, &store, 1, Some(1), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(report.results.len(), 1);
    let result = &report.results[0];
    assert!(result.triples[0].hop_alternatives.is_empty());
    assert!(result.relaxed_query.is_none());
    assert_eq!(result.row_count, 0);

    // No real relaxation was found, but the pruned fallback (culprit triple
    // just dropped, no path substitution) is still there and non-empty.
    assert!(!result.pruned_query.contains("hasSensor"), "the culprit triple should be dropped, not replaced");
    assert_eq!(result.pruned_row_count, 1, "sensor1 still matches once the broken triple is just dropped");
}

#[test]
fn pruned_query_is_present_even_when_a_real_relaxation_is_found() {
    // pruned_query isn't only emitted on failure — it's always populated,
    // even alongside a successful relaxed_query, so callers can compare
    // the two or fall back later without a second call.
    let store = test_store();
    let report = diagnose_and_relax(BROKEN_QUERY, &store, 1, Some(4), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    let result = &report.results[0];
    assert!(result.relaxed_query.is_some(), "sanity check: a real relaxation was found here");
    assert!(!result.pruned_query.contains("hasSensor"), "the culprit triple should be dropped, not replaced");
    assert_eq!(result.pruned_row_count, 1, "sensor1 is still reachable once the broken triple is just dropped");
}

#[test]
fn pruned_row_count_respects_result_limit() {
    // Dropping the culprit's hasSensor triple leaves both sensor1 and
    // sensor2 matching `?sensor a ex:TempSensor`, so pruned_row_count
    // should be tightened by result_limit the same way a real relaxed
    // query's row count is.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:buildingA ex:hasPart ex:zoneA .
                ex:zoneA ex:hasSensor ex:sensor1 .
                ex:sensor1 a ex:TempSensor .
                ex:buildingA ex:hasDevice ex:sensor2 .
                ex:sensor2 a ex:TempSensor .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:buildingA ex:hasSensor ?sensor .
            ?sensor a ex:TempSensor .
        }
    "#;

    let unbounded = diagnose_and_relax(query, &store, 1, Some(4), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(unbounded.results[0].pruned_row_count, 2);

    let capped = diagnose_and_relax(query, &store, 1, Some(4), Some(5), Some(1), NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(capped.results[0].pruned_row_count, 1, "result_limit tightens pruned_row_count too");
}

#[test]
fn falls_back_to_pruned_query_when_the_relaxation_timeout_is_exceeded() {
    // An effectively-zero timeout means the deadline is already passed by
    // the time relax_combo's first query would run, so every SPARQL
    // execution inside it degrades the same way a genuinely slow query
    // would: `relaxed_query: None`, empty hop_alternatives, no error and no
    // hang for the whole `diagnose_and_relax` call.
    let store = test_store();
    let report = diagnose_and_relax(
        BROKEN_QUERY,
        &store,
        1,
        Some(4),
        Some(5),
        None,
        NamespaceScope::Unrestricted,
        Some(Duration::from_nanos(1)),
        None,
    )
    .unwrap();
    assert_eq!(report.results.len(), 1);
    let result = &report.results[0];
    assert!(result.relaxed_query.is_none(), "no query work could complete within the timeout");
    assert!(result.triples[0].hop_alternatives.is_empty());
    assert!(!result.pruned_query.is_empty(), "the pruned fallback's text needs no store access, so it's still there");
    assert!(!result.pruned_query.contains("hasSensor"));
}

fn value_store() -> Store {
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:zone1 ex:hasSensor ex:sensor1 .
                ex:sensor1 ex:hasValue 72 .
            "#,
        )
        .unwrap();
    store
}

#[test]
fn diagnoses_an_overly_restrictive_filter_as_a_culprit() {
    let store = value_store();
    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor ?value WHERE {
            ex:zone1 ex:hasSensor ?sensor .
            ?sensor ex:hasValue ?value .
            FILTER(?value > 1000)
        }
    "#;

    let diagnosis = diagnose(query, &store, 1, None).unwrap();
    assert_eq!(diagnosis.original_row_count, 0);
    assert!(diagnosis.culprits.is_empty(), "no BGP triple is broken here, only the filter");
    assert_eq!(diagnosis.filter_culprits.len(), 1);
    assert_eq!(diagnosis.filter_culprits[0].row_count_without_filter, 1);
}

#[test]
fn does_not_flag_a_filter_that_is_not_excluding_anything() {
    let store = value_store();
    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor ?value WHERE {
            ex:zone1 ex:hasSensor ?sensor .
            ?sensor ex:hasValue ?value .
            FILTER(?value > 0)
        }
    "#;

    let diagnosis = diagnose(query, &store, 1, None).unwrap();
    assert_eq!(diagnosis.original_row_count, 1);
    assert!(diagnosis.filter_culprits.is_empty());
}

#[test]
fn diagnose_and_relax_reports_filters_without_attempting_a_relaxation() {
    let store = value_store();
    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor ?value WHERE {
            ex:zone1 ex:hasSensor ?sensor .
            ?sensor ex:hasValue ?value .
            FILTER(?value > 1000)
        }
    "#;

    let report = diagnose_and_relax(query, &store, 1, Some(4), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert!(report.results.is_empty());
    assert_eq!(report.filter_results.len(), 1);
    assert_eq!(report.filter_results[0].row_count_without_filter, 1);
    assert!(report.filter_results[0].expression_text.contains("1000"));
}

#[test]
fn combines_distinct_paths_from_different_bound_pairs_as_alternatives() {
    // buildingA reaches sensor1 only via a 2-hop path (hasPart/hasSensor)
    // and sensor2 only via an unrelated 1-hop path (hasDevice). No single
    // path connects both, so picking just one (as an earlier version of
    // this tool did) would silently drop one of the two valid sensors.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:buildingA ex:hasPart ex:zoneA .
                ex:zoneA ex:hasSensor ex:sensor1 .
                ex:sensor1 a ex:TempSensor .
                ex:buildingA ex:hasDevice ex:sensor2 .
                ex:sensor2 a ex:TempSensor .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:buildingA ex:hasSensor ?sensor .
            ?sensor a ex:TempSensor .
        }
    "#;

    let diagnosis = diagnose(query, &store, 1, None).unwrap();
    assert_eq!(diagnosis.original_row_count, 0);
    assert_eq!(diagnosis.culprits.len(), 1);

    let report = diagnose_and_relax(query, &store, 1, Some(4), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(report.results.len(), 1);
    let result = &report.results[0];

    assert_eq!(
        result.triples[0].hop_alternatives.len(),
        2,
        "both distinct paths should be kept, not just one"
    );
    assert_eq!(
        result.row_count, 2,
        "the relaxed query should recover both sensor1 (via hasPart/hasSensor) and sensor2 (via hasDevice)"
    );
}

#[test]
fn reuses_one_path_across_endpoints_that_share_its_shape() {
    // Both sensors hang off buildingA via the same hasPart/hasSensor shape
    // (through different intermediate zones), so the hop sequence found for
    // the first sampled endpoint should generalize to the second via
    // `path_holds` rather than needing its own independent BFS — and either
    // way, both sensors must still come back in the relaxed query's rows.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:buildingA ex:hasPart ex:zoneA .
                ex:zoneA ex:hasSensor ex:sensor1 .
                ex:sensor1 a ex:TempSensor .
                ex:buildingA ex:hasPart ex:zoneB .
                ex:zoneB ex:hasSensor ex:sensor2 .
                ex:sensor2 a ex:TempSensor .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:buildingA ex:hasSensor ?sensor .
            ?sensor a ex:TempSensor .
        }
    "#;

    let report = diagnose_and_relax(query, &store, 1, Some(4), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(report.results.len(), 1);
    let result = &report.results[0];

    assert_eq!(result.triples[0].hop_alternatives.len(), 1, "one shared path shape should be deduplicated, not repeated per endpoint");
    assert_eq!(result.triples[0].path_text.as_deref(), Some("(<urn:example#hasPart> / <urn:example#hasSensor>)"));
    assert_eq!(result.row_count, 2, "the single generalized path should still recover both sensor1 and sensor2");
}

#[test]
fn sample_limit_none_samples_every_distinct_bound_pair() {
    // Six sensors, each reachable from the building only via its own
    // distinct 1-hop predicate. A capped sample_limit=5 would only see the
    // first 5 in whatever order the query engine returns them; None must
    // see all 6.
    let mut ttl = String::from("@prefix ex: <urn:example#> .\n");
    for i in 1..=6 {
        ttl.push_str(&format!("ex:buildingA ex:hasDevice{i} ex:sensor{i} .\n"));
        ttl.push_str(&format!("ex:sensor{i} a ex:TempSensor .\n"));
    }
    let store = Store::new().unwrap();
    store.load_from_slice(RdfParser::from_format(RdfFormat::Turtle), &ttl).unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:buildingA ex:hasSensor ?sensor .
            ?sensor a ex:TempSensor .
        }
    "#;

    let capped = diagnose_and_relax(query, &store, 1, Some(2), Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(capped.results[0].triples[0].hop_alternatives.len(), 5, "capped sampling stops at the limit");

    let uncapped = diagnose_and_relax(query, &store, 1, Some(2), None, None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(
        uncapped.results[0].triples[0].hop_alternatives.len(),
        6,
        "None samples every distinct pair"
    );
    assert_eq!(uncapped.results[0].row_count, 6, "the relaxed query recovers all six sensors");
}

#[test]
fn depth_1_finds_nothing_but_depth_2_finds_a_joint_two_triple_culprit() {
    // Building reaches sensor1 only via hasZone/hasSensor (2 hops), and
    // sensor1's unit is Celsius, not Fahrenheit. The query below has TWO
    // wrong triples at once: a direct (nonexistent) hasSensor edge, and a
    // wrong unit. Removing either ONE alone still leaves the other broken,
    // so depth 1 finds nothing; only removing BOTH together unblocks the
    // query, which depth 2 should find.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:building ex:hasZone ex:zone1 .
                ex:zone1 ex:hasSensor ex:sensor1 .
                ex:sensor1 a ex:TempSensor .
                ex:sensor1 ex:hasUnit ex:Celsius .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:building ex:hasSensor ?sensor .
            ?sensor ex:hasUnit ex:Fahrenheit .
            ?sensor a ex:TempSensor .
        }
    "#;

    let depth_1 = diagnose(query, &store, 1, None).unwrap();
    assert!(depth_1.culprits.is_empty(), "no single triple's removal alone unblocks the query");

    let depth_3 = diagnose(query, &store, 3, None).unwrap();
    assert_eq!(depth_3.culprits.len(), 1, "exactly one joint 2-triple culprit should be found");
    let culprit = &depth_3.culprits[0];
    assert_eq!(culprit.depth, 2, "found at depth 2, so depth 3 never had to run");
    assert_eq!(culprit.triples.len(), 2);
    let predicates: Vec<String> = culprit.triples.iter().map(|t| t.predicate.to_string()).collect();
    assert!(predicates.contains(&"<urn:example#hasSensor>".to_string()));
    assert!(predicates.contains(&"<urn:example#hasUnit>".to_string()));

    // Relaxation: the hasSensor triple has a real path (hasZone/hasSensor),
    // but the hasUnit triple genuinely has none (nothing connects sensor1
    // to Fahrenheit). Since at least one triple in the pair found a path,
    // the combination should still be relaxed as a whole: hasSensor spliced
    // in with its discovered path, hasUnit simply dropped — recovering
    // sensor1 rather than giving up on the pair entirely.
    let report = diagnose_and_relax(query, &store, 3, Some(4), None, None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(report.results.len(), 1);
    let result = &report.results[0];
    assert_eq!(result.triples.len(), 2);
    let has_sensor_result =
        result.triples.iter().find(|t| t.triple_text.contains("hasSensor")).unwrap();
    let has_unit_result = result.triples.iter().find(|t| t.triple_text.contains("hasUnit")).unwrap();
    assert!(!has_sensor_result.hop_alternatives.is_empty(), "hasSensor has a real 2-hop path");
    assert!(has_unit_result.hop_alternatives.is_empty(), "nothing connects sensor1 to Fahrenheit");
    assert!(
        result.relaxed_query.is_some(),
        "hasSensor's path was found, so the pair is still relaxed with hasUnit simply dropped"
    );
    assert!(!result.relaxed_query.as_ref().unwrap().contains("hasUnit"), "hasUnit should be dropped, not just left broken");
    assert_eq!(result.row_count, 1, "sensor1 matches once hasSensor is path-substituted and hasUnit is dropped");
}

#[test]
fn does_not_relax_when_the_object_is_unconstrained_elsewhere() {
    // `?sensor` appears *only* in the broken triple below, so once it's
    // removed to check ablation, nothing else binds it — there's no
    // specific target to search for, only the subject (a concrete IRI).
    // A real 1-hop path (hasZone) does exist from `building`, but a
    // one-sided endpoint isn't searched at all (see the module docs), so no
    // relaxation should be attempted despite that real path existing.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:building ex:hasZone ex:zone1 .
                ex:zone1 ex:hasSensor ex:sensor1 .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:building ex:hasSensor ?sensor .
        }
    "#;

    let diagnosis = diagnose(query, &store, 1, None).unwrap();
    assert_eq!(diagnosis.culprits.len(), 1, "building has no direct hasSensor edge");

    let report = diagnose_and_relax(query, &store, 1, Some(2), None, None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(report.results.len(), 1);
    let relaxed_triple = &report.results[0].triples[0];
    assert!(relaxed_triple.hop_alternatives.is_empty(), "one-sided endpoints aren't searched");
    assert!(relaxed_triple.path_text.is_none());
    assert!(report.results[0].relaxed_query.is_none());
    assert_eq!(report.results[0].row_count, 0);
}

#[test]
fn does_not_relax_when_the_subject_is_unconstrained_elsewhere() {
    // `?x` appears only in the broken triple, and this time it's the
    // *subject* that's unconstrained, not the object. A real path
    // (partOf) does exist, but again, a one-sided endpoint isn't searched.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:sensor1 ex:partOf ex:building .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?x WHERE {
            ?x ex:hasSensor ex:building .
        }
    "#;

    let report = diagnose_and_relax(query, &store, 1, Some(1), None, None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(report.results.len(), 1);
    let relaxed_triple = &report.results[0].triples[0];
    assert!(relaxed_triple.hop_alternatives.is_empty(), "one-sided endpoints aren't searched");
    assert!(report.results[0].relaxed_query.is_none());
    assert_eq!(report.results[0].row_count, 0);
}

#[test]
fn default_ablation_depth_escalates_on_its_own_to_find_a_joint_culprit() {
    // Same two-simultaneously-wrong-triples scenario as
    // `depth_1_finds_nothing_but_depth_2_finds_a_joint_two_triple_culprit`,
    // but via the all-defaults entry point: ablation_depth defaults to 3,
    // so it should escalate past depth 1 without being told to.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:building ex:hasZone ex:zone1 .
                ex:zone1 ex:hasSensor ex:sensor1 .
                ex:sensor1 a ex:TempSensor .
                ex:sensor1 ex:hasUnit ex:Celsius .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:building ex:hasSensor ?sensor .
            ?sensor ex:hasUnit ex:Fahrenheit .
            ?sensor a ex:TempSensor .
        }
    "#;

    let report = diagnose_and_relax_default(query, &store).unwrap();
    assert_eq!(report.results.len(), 1, "default ablation_depth=3 should escalate to find the joint culprit");
    assert_eq!(report.results[0].found_at_depth, 2);
}

#[test]
fn default_max_depth_is_2_for_pair_search() {
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:building ex:hasZone ex:zone1 .
                ex:zone1 ex:hasSensor ex:sensor1 .
                ex:sensor1 a ex:TempSensor .
            "#,
        )
        .unwrap();

    // Both sides bound: the 2-hop hasZone/hasSensor path should be found by
    // default (DEFAULT_PAIR_SEARCH_DEPTH = 2).
    let pair_query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:building ex:hasSensor ?sensor .
            ?sensor a ex:TempSensor .
        }
    "#;
    let pair_report = diagnose_and_relax(pair_query, &store, 1, None, Some(5), None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(pair_report.results[0].row_count, 1, "default pair-search depth (2) finds the 2-hop path");
}

#[test]
fn namespace_scope_restricts_path_search_to_allowed_prefixes() {
    // The only real edge connecting `building` to `zone1` uses a predicate
    // outside the Brick/S223/RDFS/QUDT namespaces `NamespaceScope::default`
    // restricts to, so a namespace-restricted search should find nothing
    // even though an unrestricted search finds it easily.
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:building ex:altPath ex:zone1 .
                ex:zone1 a ex:Zone .
            "#,
        )
        .unwrap();

    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?zone WHERE {
            ex:building ex:hasZone ?zone .
            ?zone a ex:Zone .
        }
    "#;

    let unrestricted = diagnose_and_relax(query, &store, 1, Some(1), None, None, NamespaceScope::Unrestricted, None, None).unwrap();
    assert_eq!(unrestricted.results[0].triples[0].hop_alternatives.len(), 1, "the real altPath edge should be found");
    assert_eq!(unrestricted.results[0].row_count, 1);

    let brick_restricted = diagnose_and_relax(query, &store, 1, Some(1), None, None, NamespaceScope::default(), None, None).unwrap();
    assert!(
        brick_restricted.results[0].triples[0].hop_alternatives.is_empty(),
        "ex:altPath isn't in any allowed namespace, so it shouldn't be found"
    );
    assert!(brick_restricted.results[0].relaxed_query.is_none());

    let custom_restricted = diagnose_and_relax(
        query,
        &store,
        1,
        Some(1),
        None,
        None,
        NamespaceScope::Only(vec!["urn:example#alt".to_string()]),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        custom_restricted.results[0].triples[0].hop_alternatives.len(),
        1,
        "a caller-supplied namespace covering ex:altPath should find it"
    );
}

/// `?sensor` (T1/T2) and `?widget` (T3) never share a variable, so removing
/// either T1 or T2 alone leaves a disconnected two-component BGP — exactly
/// the shape that forces a cartesian product. T3's own component (`ex:Widget`)
/// is non-empty, so if this weren't guarded, checking those combos would ask
/// the query engine to materialize a real (if here, trivially small) cross
/// product; at real-world scale this is the class of query that can make an
/// engine block on a full N×M materialization well past any `timeout` (see
/// `diagnose.rs`'s module docs).
fn disconnected_store() -> Store {
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:building223 ex:hasSensor ex:sensor1 .
                ex:sensor1 a ex:TempSensor .
                ex:widget1 a ex:Widget .
            "#,
        )
        .unwrap();
    store
}

const DISCONNECTED_QUERY: &str = r#"
    PREFIX ex: <urn:example#>
    SELECT ?sensor ?widget WHERE {
        ex:building223 ex:hasBrokenLink ?sensor .
        ?sensor a ex:TempSensor .
        ?widget a ex:Widget .
    }
"#;

#[test]
fn a_disconnected_reduced_pattern_is_reported_as_a_cartesian_risk_not_silently_ruled_out() {
    let store = disconnected_store();
    let diagnosis = diagnose(DISCONNECTED_QUERY, &store, 1, None).unwrap();

    assert_eq!(diagnosis.original_row_count, 0);
    // Removing the real culprit (T1, `ex:hasBrokenLink`) leaves {T2, T3}
    // disconnected, and removing T2 leaves {T1, T3} disconnected too — both
    // are never evaluated, so the real culprit is never confirmed as one.
    // That's the conservative trade-off this guard makes deliberately (see
    // `ComboVerdict`'s docs): a query never evaluated can't be claimed
    // "not a culprit" any more than it can be claimed a culprit.
    assert!(diagnosis.culprits.is_empty(), "the true culprit's combo is disconnected once isolated, so it's never confirmed — that's expected, not a bug");
    assert_eq!(diagnosis.cartesian_risks.len(), 2, "both single-triple combos that disconnect the pattern should be reported as risks");

    // Removing T3 alone leaves {T1, T2} connected (they share ?sensor), so
    // that combo *is* safely evaluated — and correctly comes back as not a
    // culprit, since T1 is still broken within it.
    let risky_triples: Vec<String> = diagnosis.cartesian_risks.iter().flat_map(|r| r.triples.iter().map(ToString::to_string)).collect();
    assert!(!risky_triples.iter().any(|t| t.contains("Widget")), "the T3-removed combo stays connected and is safely evaluated, not flagged as a risk");
}

#[test]
fn diagnose_and_relax_does_not_hang_or_error_on_a_disconnected_query() {
    let store = disconnected_store();
    let report = diagnose_and_relax(DISCONNECTED_QUERY, &store, 1, None, None, None, NamespaceScope::Unrestricted, None, None).unwrap();
    // No culprit was ever confirmed (see the diagnosis-only test above), so
    // there's nothing for the relax phase to even attempt.
    assert!(report.results.is_empty());
}

/// Same shape as `disconnected_store`/`DISCONNECTED_QUERY` (`?sensor` and
/// `?widget` never share a variable in the required pattern once `T1` is
/// removed), except an `OPTIONAL` triple also directly mentions both
/// `?sensor` and `?widget`. An `OPTIONAL` match can be entirely absent
/// without eliminating the outer row, so that shared pair must not count as
/// connecting them for cartesian-risk purposes — the *required* pattern is
/// still disconnected regardless of whether the optional triple happens to
/// bind for a given solution.
fn disconnected_with_optional_bridge_store() -> Store {
    let store = Store::new().unwrap();
    store
        .load_from_slice(
            RdfParser::from_format(RdfFormat::Turtle),
            r#"
                @prefix ex: <urn:example#> .
                ex:building223 ex:hasSensor ex:sensor1 .
                ex:sensor1 a ex:TempSensor .
                ex:widget1 a ex:Widget .
            "#,
        )
        .unwrap();
    store
}

const DISCONNECTED_WITH_OPTIONAL_BRIDGE_QUERY: &str = r#"
    PREFIX ex: <urn:example#>
    SELECT ?sensor ?widget WHERE {
        ex:building223 ex:hasBrokenLink ?sensor .
        ?sensor a ex:TempSensor .
        ?widget a ex:Widget .
        OPTIONAL { ?sensor ex:relatesToWidget ?widget . }
    }
"#;

#[test]
fn an_optional_only_shared_variable_does_not_mask_a_cartesian_risk_in_the_required_pattern() {
    let store = disconnected_with_optional_bridge_store();
    let diagnosis = diagnose(DISCONNECTED_WITH_OPTIONAL_BRIDGE_QUERY, &store, 1, None).unwrap();

    assert_eq!(diagnosis.original_row_count, 0);
    // Removing the real culprit (`ex:hasBrokenLink`) leaves `?sensor` and
    // `?widget` disconnected in the required pattern; the `OPTIONAL` triple
    // mentions both, but must not be treated as bridging them for this
    // check (see the function docs above). Without the fix, this combo's
    // reduced pattern (still carrying the OPTIONAL triple) looks connected,
    // gets evaluated, and — since `?sensor1 a ex:TempSensor` and `?widget1 a
    // ex:Widget` both hold regardless of the OPTIONAL — is wrongly confirmed
    // as a culprit.
    assert!(
        diagnosis.culprits.is_empty(),
        "the OPTIONAL-only shared variable must not mask the required pattern's disconnect: this combo should never be evaluated, so it can't be confirmed a culprit"
    );
    // Risky combos: removing T1 (`hasBrokenLink`) or T2 (`?sensor a
    // TempSensor`) each leave the required pattern disconnected across
    // {?sensor} vs {?widget}; removing the OPTIONAL triple itself leaves the
    // required pattern (T1, T2, T3) untouched, which is disconnected too
    // (`?widget` never appears in a required triple). Only removing T3
    // (`?widget a Widget`) leaves a connected required pattern ({T1, T2}
    // share `?sensor`), so that's the one combo actually evaluated.
    assert_eq!(diagnosis.cartesian_risks.len(), 3, "every ablation candidate except removing the ?widget triple should disconnect the required pattern");
}
