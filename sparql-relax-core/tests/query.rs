use oxigraph::io::{RdfFormat, RdfParser};
use oxigraph::store::Store;
use sparql_relax_core::query::{QueryOutcome, RdfTerm};
use sparql_relax_core::{RelaxError, query};
use std::time::Duration;

const TTL: &str = r#"
    @prefix ex: <urn:example#> .
    ex:building223 ex:hasPart ex:zone1 .
    ex:zone1 ex:hasSensor ex:sensor1 .
    ex:sensor1 a ex:TempSensor .
    ex:sensor2 a ex:TempSensor .
    ex:sensor3 a ex:TempSensor .
"#;

fn test_store() -> Store {
    let store = Store::new().unwrap();
    store.load_from_slice(RdfParser::from_format(RdfFormat::Turtle), TTL).unwrap();
    store
}

#[test]
fn select_returns_variables_and_rows() {
    let store = test_store();
    let outcome = query("PREFIX ex: <urn:example#> SELECT ?s WHERE { ?s a ex:TempSensor }", &store, None, None).unwrap();
    let QueryOutcome::Solutions { variables, rows } = outcome else { panic!("expected Solutions") };
    assert_eq!(variables, vec!["s"]);
    assert_eq!(rows.len(), 3);
    for row in &rows {
        assert_eq!(row.len(), 1);
        assert!(matches!(&row[0], Some(RdfTerm::Iri(_))));
    }
}

#[test]
fn select_leaves_unbound_optional_variables_as_none() {
    let store = test_store();
    let outcome = query(
        "PREFIX ex: <urn:example#> SELECT ?s ?zone WHERE { ?s a ex:TempSensor . OPTIONAL { ?zone ex:hasSensor ?s } }",
        &store,
        None,
        None,
    )
    .unwrap();
    let QueryOutcome::Solutions { variables, rows } = outcome else { panic!("expected Solutions") };
    assert_eq!(variables, vec!["s", "zone"]);
    // sensor1 has a binding for ?zone (zone1 hasSensor sensor1); sensor2/sensor3 don't.
    let unbound_zone_rows = rows.iter().filter(|row| row[1].is_none()).count();
    assert_eq!(unbound_zone_rows, 2);
}

#[test]
fn ask_returns_boolean() {
    let store = test_store();
    let outcome = query("PREFIX ex: <urn:example#> ASK { ex:sensor1 a ex:TempSensor }", &store, None, None).unwrap();
    assert_eq!(outcome, QueryOutcome::Boolean(true));

    let outcome = query("PREFIX ex: <urn:example#> ASK { ex:sensor1 a ex:HumiditySensor }", &store, None, None).unwrap();
    assert_eq!(outcome, QueryOutcome::Boolean(false));
}

#[test]
fn construct_returns_a_graph() {
    let store = test_store();
    let outcome =
        query("PREFIX ex: <urn:example#> CONSTRUCT { ?s a ex:Thing } WHERE { ?s a ex:TempSensor }", &store, None, None).unwrap();
    let QueryOutcome::Graph(triples) = outcome else { panic!("expected Graph") };
    assert_eq!(triples.len(), 3);
    assert!(triples.iter().all(|t| t.predicate == RdfTerm::Iri("http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string())));
}

#[test]
fn row_limit_tightens_an_injected_limit() {
    let store = test_store();
    let outcome = query("PREFIX ex: <urn:example#> SELECT ?s WHERE { ?s a ex:TempSensor }", &store, Some(2), None).unwrap();
    let QueryOutcome::Solutions { rows, .. } = outcome else { panic!("expected Solutions") };
    assert_eq!(rows.len(), 2);
}

#[test]
fn row_limit_only_tightens_an_existing_limit_never_loosens_it() {
    let store = test_store();
    let outcome =
        query("PREFIX ex: <urn:example#> SELECT ?s WHERE { ?s a ex:TempSensor } LIMIT 1", &store, Some(50), None).unwrap();
    let QueryOutcome::Solutions { rows, .. } = outcome else { panic!("expected Solutions") };
    assert_eq!(rows.len(), 1);
}

#[test]
fn row_limit_has_no_effect_on_ask() {
    let store = test_store();
    let outcome = query("PREFIX ex: <urn:example#> ASK { ex:sensor1 a ex:TempSensor }", &store, Some(1), None).unwrap();
    assert_eq!(outcome, QueryOutcome::Boolean(true));
}

#[test]
fn a_past_deadline_returns_query_timeout_rather_than_running() {
    let store = test_store();
    let result = query("PREFIX ex: <urn:example#> SELECT ?s WHERE { ?s a ex:TempSensor }", &store, None, Some(Duration::from_nanos(1)));
    assert!(matches!(result, Err(RelaxError::QueryTimeout)), "expected a QueryTimeout error");
}

#[test]
fn syntax_errors_propagate() {
    let store = test_store();
    let result = query("SELECT ?s WHERE { ?s", &store, None, None);
    assert!(matches!(result, Err(RelaxError::Syntax(_))), "expected a Syntax error");
}
