import sparql_relax
import pyoxigraph

# Load a small graph
text = "prefix brick: <https://brickschema.org/schema/Brick#> brick:Zone a brick:Zone ."
store = sparql_relax.Store(text)
query = "SELECT ?z WHERE { ?z a brick:Something }"
print("Diagnosing...")
try:
    report = store.diagnose_and_relax(query, timeout=5.0, diagnose_timeout=5.0)
    print("Report:", report)
except Exception as e:
    print("Error:", e)
