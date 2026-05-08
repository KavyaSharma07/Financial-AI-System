from src.database import create_tables

print("Connecting to PostgreSQL...")
create_tables()
print("Phase 2 complete — database is ready.")