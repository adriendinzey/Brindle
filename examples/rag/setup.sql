-- Brindle RAG example — schema, corpus, and indexes.
--
-- Run this once against a database with the Brindle extension available:
--     \i setup.sql
-- then work through queries.sql. Re-running it is safe: it drops and rebuilds.
--
-- The corpus is 16 products across four categories. Each row carries:
--   * a text description        -> the lexical (full-text) signal,
--   * a brindle_vector embedding -> the semantic signal,
--   * category_id / price / rating / in_stock -> structured filters.
-- The embeddings are hand-authored topic vectors (see embed.py and README.md),
-- so the whole walkthrough is deterministic with no model download or API key.

CREATE EXTENSION IF NOT EXISTS brindle;

DROP TABLE IF EXISTS products;

CREATE TABLE products (
    id          bigint PRIMARY KEY,
    name        text        NOT NULL,
    category_id int         NOT NULL,   -- 1=Audio  2=Outdoor  3=Kitchen  4=Office
    price       float8      NOT NULL,
    rating      real        NOT NULL,
    in_stock    boolean     NOT NULL,
    description text        NOT NULL,
    embedding   brindle_vector,         -- 8-dim topic vector; see embed.py
    -- The lexical signal, kept in sync with the text by the database. brindle_hybrid
    -- can also lex a plain text column on the fly, but a stored tsvector lets a GIN
    -- index serve the match.
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', name || ' ' || description)) STORED
);

INSERT INTO products (id, name, category_id, price, rating, in_stock, description, embedding) VALUES
  (1, 'Studio Wireless Headphones', 1, 89.99, 4.6, true, 'Closed-back wireless headphones tuned for focused studio listening at a desk.', '[0.9,0.3,0.0,0.0,0.0,0.6,0.0,0.7]'),
  (2, 'Reference Studio Monitors', 1, 149.0, 4.8, true, 'Over-ear studio monitors with a flat response and Bluetooth for critical desk mixing.', '[0.95,0.1,0.0,0.0,0.0,0.6,0.0,0.5]'),
  (3, 'Sport Wireless Earbuds', 1, 59.99, 4.3, true, 'Sweatproof wireless earbuds that stay put during runs and gym workouts.', '[0.7,0.8,0.1,0.5,0.0,0.0,0.8,0.7]'),
  (4, 'Noise-Cancelling Earbuds', 1, 129.0, 4.5, true, 'Active noise-cancelling earbuds with a wireless charging case for commutes.', '[0.85,0.7,0.0,0.0,0.0,0.2,0.1,0.8]'),
  (5, 'Waterproof Shower Speaker', 1, 24.99, 4.1, true, 'Compact waterproof Bluetooth speaker with a suction mount for the shower.', '[0.7,0.0,0.3,0.9,0.0,0.0,0.0,0.6]'),
  (6, 'Trail Running Vest', 2, 74.99, 4.4, true, 'Lightweight hydration vest for trail running with a weatherproof shell.', '[0.0,0.7,0.8,0.6,0.0,0.0,0.8,0.0]'),
  (7, 'Waterproof Hiking Jacket', 2, 159.0, 4.7, true, 'Breathable waterproof shell jacket for hiking in heavy rain.', '[0.0,0.6,0.9,0.95,0.0,0.0,0.2,0.0]'),
  (8, 'Rugged Wireless Headphones', 2, 45.0, 3.9, true, 'Rugged wireless headphones with a weatherproof, sweat-resistant build for the outdoors.', '[0.5,0.4,0.8,0.7,0.0,0.0,0.3,0.6]'),
  (9, 'Insulated Water Bottle', 2, 19.99, 4.2, true, 'Vacuum-insulated stainless steel water bottle that keeps drinks cold on the trail.', '[0.0,0.4,0.7,0.8,0.3,0.0,0.5,0.0]'),
  (10, 'Camping Lantern', 2, 34.99, 4.0, false, 'Rechargeable LED camping lantern with a collapsible, weather-resistant body.', '[0.0,0.1,0.9,0.3,0.0,0.1,0.0,0.2]'),
  (11, 'Chef''s Knife', 3, 79.0, 4.9, true, 'Forged high-carbon chef''s knife balanced for everyday kitchen prep.', '[0.0,0.0,0.0,0.0,0.95,0.1,0.0,0.0]'),
  (12, 'Espresso Machine', 3, 249.0, 4.6, true, 'Compact espresso machine with a steam wand for lattes at home.', '[0.0,0.0,0.0,0.2,0.9,0.1,0.0,0.1]'),
  (13, 'Cast-Iron Skillet', 3, 39.99, 4.7, true, 'Pre-seasoned cast-iron skillet that moves from stovetop to oven to campfire.', '[0.0,0.0,0.2,0.0,0.9,0.0,0.0,0.0]'),
  (14, 'Standing Desk', 4, 329.0, 4.5, true, 'Electric height-adjustable standing desk for a healthier home office.', '[0.0,0.0,0.0,0.0,0.0,0.95,0.2,0.0]'),
  (15, 'Ergonomic Office Chair', 4, 199.0, 4.4, true, 'Mesh-back ergonomic office chair with lumbar support for long work sessions.', '[0.0,0.2,0.0,0.0,0.0,0.9,0.3,0.0]'),
  (16, 'Wireless Mechanical Keyboard', 4, 99.0, 4.5, true, 'Low-profile wireless mechanical keyboard for a tidy desk setup.', '[0.0,0.1,0.0,0.0,0.0,0.8,0.0,0.7]');

-- The Brindle index. The vector column comes first with a metric operator class
-- (cosine here, matched by the <=> operator in queries.sql). The filter columns
-- follow as KEY columns, NOT INCLUDE columns: only a key column's qual reaches
-- the graph traversal, which is what keeps recall high under a selective WHERE.
-- All four filter types Brindle supports are here to experiment with — integer
-- (category_id), float (price, rating) and boolean (in_stock). See docs/FILTERING.md.
CREATE INDEX products_embedding_idx
    ON products USING brindle (embedding brindle_vector_cosine_ops, category_id, price, rating, in_stock);

-- Serves the lexical side of the hybrid search.
CREATE INDEX products_tsv_idx ON products USING gin (tsv);

ANALYZE products;
