-- MASTER SCHEMA RUNNER (HAIP)
-- Rebuilds the full medallion architecture in order:
--   bronze -> silver -> gold
--
-- Drops gold first (reverse order) because gold views
-- depend on silver tables.
--
-- Usage (from project root):
--   sudo -u postgres psql -d haip < database/schema.sql

\echo 'Dropping dependent layers (gold) first...'
DROP SCHEMA IF EXISTS gold CASCADE;

\echo 'Building BRONZE layer...'
\i database/bronze_tables.sql

\echo 'Building SILVER layer...'
\i database/silver_tables.sql

\echo 'Building GOLD layer...'
\i database/gold_views.sql

\echo 'Medallion schema build complete.'
