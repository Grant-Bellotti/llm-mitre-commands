import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.json as pj

# Read JSONL
table = pj.read_json("mitre_techniques_descriptions.jsonl")   # auto-detects JSON lines

# Write to Parquet
pq.write_table(table, "data.parquet")
