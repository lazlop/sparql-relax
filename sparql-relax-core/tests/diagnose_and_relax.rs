use oxigraph::io::{RdfFormat, RdfParser};
use oxigraph::store::Store;
use sparql_relax_core::bfs::Hop;
use sparql_relax_core::{diagnose, diagnose_and_relax, diagnose_and_relax_default};

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
    let diagnosis = diagnose(BROKEN_QUERY, &store, 1).unwrap();

    assert_eq!(diagnosis.original_row_count, 0);
    assert_eq!(diagnosis.culprits.len(), 1);
    let culprit = &diagnosis.culprits[0];
    assert_eq!(culprit.depth, 1);
    assert_eq!(culprit.triples.len(), 1);
    assert_eq!(culprit.triples[0].predicate.to_string(), "<urn:example#hasSensor>");
}

#[test]
fn finds_a_real_forward_forward_path_and_fixes_the_query() {
    let store = test_store();
    let report = diagnose_and_relax(BROKEN_QUERY, &store, 1, Some(4), Some(5)).unwrap();

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
fn query_with_no_broken_triples_reports_no_culprits() {
    let store = test_store();
    let query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:zone1 ex:hasSensor ?sensor .
            ?sensor a ex:TempSensor .
        }
    "#;
    let diagnosis = diagnose(query, &store, 1).unwrap();
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

    let report = diagnose_and_relax(query, &store, 1, Some(4), Some(5)).unwrap();
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
    let report = diagnose_and_relax(BROKEN_QUERY, &store, 1, Some(1), Some(5)).unwrap();
    assert_eq!(report.results.len(), 1);
    let result = &report.results[0];
    assert!(result.triples[0].hop_alternatives.is_empty());
    assert!(result.relaxed_query.is_none());
    assert_eq!(result.row_count, 0);
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

    let diagnosis = diagnose(query, &store, 1).unwrap();
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

    let diagnosis = diagnose(query, &store, 1).unwrap();
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

    let report = diagnose_and_relax(query, &store, 1, Some(4), Some(5)).unwrap();
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

    let diagnosis = diagnose(query, &store, 1).unwrap();
    assert_eq!(diagnosis.original_row_count, 0);
    assert_eq!(diagnosis.culprits.len(), 1);

    let report = diagnose_and_relax(query, &store, 1, Some(4), Some(5)).unwrap();
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

    let capped = diagnose_and_relax(query, &store, 1, Some(2), Some(5)).unwrap();
    assert_eq!(capped.results[0].triples[0].hop_alternatives.len(), 5, "capped sampling stops at the limit");

    let uncapped = diagnose_and_relax(query, &store, 1, Some(2), None).unwrap();
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

    let depth_1 = diagnose(query, &store, 1).unwrap();
    assert!(depth_1.culprits.is_empty(), "no single triple's removal alone unblocks the query");

    let depth_3 = diagnose(query, &store, 3).unwrap();
    assert_eq!(depth_3.culprits.len(), 1, "exactly one joint 2-triple culprit should be found");
    let culprit = &depth_3.culprits[0];
    assert_eq!(culprit.depth, 2, "found at depth 2, so depth 3 never had to run");
    assert_eq!(culprit.triples.len(), 2);
    let predicates: Vec<String> = culprit.triples.iter().map(|t| t.predicate.to_string()).collect();
    assert!(predicates.contains(&"<urn:example#hasSensor>".to_string()));
    assert!(predicates.contains(&"<urn:example#hasUnit>".to_string()));

    // Relaxation: the hasSensor triple has a real path (hasZone/hasSensor),
    // but the hasUnit triple genuinely has none (nothing connects sensor1
    // to Fahrenheit), so the combination as a whole should not be relaxed.
    let report = diagnose_and_relax(query, &store, 3, Some(4), None).unwrap();
    assert_eq!(report.results.len(), 1);
    let result = &report.results[0];
    assert_eq!(result.triples.len(), 2);
    let has_sensor_result =
        result.triples.iter().find(|t| t.triple_text.contains("hasSensor")).unwrap();
    let has_unit_result = result.triples.iter().find(|t| t.triple_text.contains("hasUnit")).unwrap();
    assert!(!has_sensor_result.hop_alternatives.is_empty(), "hasSensor has a real 2-hop path");
    assert!(has_unit_result.hop_alternatives.is_empty(), "nothing connects sensor1 to Fahrenheit");
    assert!(result.relaxed_query.is_none(), "one broken triple in the pair has no path, so no combined fix");
    assert_eq!(result.row_count, 0);
}

#[test]
fn explores_outward_from_a_subject_only_anchor_when_the_object_is_unconstrained() {
    // `?sensor` appears *only* in the broken triple below, so once it's
    // removed to check ablation, nothing else binds it — there's no
    // specific target to search for, only the subject (a concrete IRI) to
    // explore outward from. Both a real 1-hop path (hasZone) and a real
    // 2-hop path (hasZone/hasSensor) exist from `building`, and since
    // there's no fixed goal to prefer one over the other, both should come
    // back as alternatives.
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

    let diagnosis = diagnose(query, &store, 1).unwrap();
    assert_eq!(diagnosis.culprits.len(), 1, "building has no direct hasSensor edge");

    let report = diagnose_and_relax(query, &store, 1, Some(2), None).unwrap();
    assert_eq!(report.results.len(), 1);
    let relaxed_triple = &report.results[0].triples[0];
    assert_eq!(
        relaxed_triple.hop_alternatives.len(),
        2,
        "both the 1-hop (hasZone) and 2-hop (hasZone/hasSensor) real paths should be offered"
    );
    assert_eq!(
        report.results[0].row_count,
        2,
        "the relaxed query should return both zone1 and sensor1 as suggestions"
    );
}

#[test]
fn explores_outward_from_an_object_only_anchor_and_reverses_the_path() {
    // `?x` appears only in the broken triple, and this time it's the
    // *subject* that's unconstrained, not the object — so search has to
    // anchor on the object side and flip the discovered path around to
    // still read subject -> object.
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

    let report = diagnose_and_relax(query, &store, 1, Some(1), None).unwrap();
    assert_eq!(report.results.len(), 1);
    let relaxed_triple = &report.results[0].triples[0];
    assert_eq!(relaxed_triple.path_text.as_deref(), Some("<urn:example#partOf>"));
    assert_eq!(report.results[0].row_count, 1, "sensor1 is recovered via the real partOf edge");
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
fn default_max_depth_is_adaptive_2_for_point_to_point_1_for_anchor_only() {
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

    // Point-to-point (both sides bound): the 2-hop hasZone/hasSensor path
    // should be found by default (DEFAULT_PAIR_SEARCH_DEPTH = 2).
    let pair_query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:building ex:hasSensor ?sensor .
            ?sensor a ex:TempSensor .
        }
    "#;
    let pair_report = diagnose_and_relax(pair_query, &store, 1, None, Some(5)).unwrap();
    assert_eq!(pair_report.results[0].row_count, 1, "default pair-search depth (2) finds the 2-hop path");

    // Anchor-only (only the subject is bound; `?sensor` is unconstrained
    // elsewhere): only the 1-hop hasZone path should be found by default
    // (DEFAULT_ANCHOR_SEARCH_DEPTH = 1), not the 2-hop hasZone/hasSensor one.
    let anchor_query = r#"
        PREFIX ex: <urn:example#>
        SELECT ?sensor WHERE {
            ex:building ex:hasSensor ?sensor .
        }
    "#;
    let anchor_report = diagnose_and_relax(anchor_query, &store, 1, None, Some(5)).unwrap();
    let relaxed_triple = &anchor_report.results[0].triples[0];
    assert_eq!(
        relaxed_triple.hop_alternatives.len(),
        1,
        "only the 1-hop path should be found at the default anchor-only depth"
    );
    assert_eq!(anchor_report.results[0].row_count, 1, "only zone1 should come back, not sensor1 too");
}
