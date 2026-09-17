#!/usr/bin/env python3
"""Generate the product corpus and its embeddings for the Brindle RAG example.

The embeddings here are *hand-authored topic vectors*, not the output of a neural
model. Each dimension is a named concept ("audio", "outdoor", ...); a product's
vector is how strongly it loads on each concept. Cosine distance over that space
then behaves like semantic similarity — near vectors mean related products — while
staying small, deterministic, and dependency-free, so the walkthrough's results
are reproducible on any machine with no model download and no API key.

Run it to regenerate the INSERT block in `setup.sql`:

    python3 embed.py > /tmp/rows.sql

To use *real* embeddings instead, replace `embed()` with a call to an embedding
model (e.g. sentence-transformers) and widen the `brindle_vector` column to that
model's dimensionality — the SQL and queries are otherwise unchanged. That step
is optional; the checked-in vectors above make the example self-contained.
"""

# The semantic axes. A product's embedding is its weight on each, in this order.
AXES = ["audio", "wearable", "outdoor", "water", "kitchen", "office", "fitness", "wireless"]

# id: (name, category_id, price, rating, in_stock, {axis: weight}, description)
# category_id legend: 1=Audio, 2=Outdoor, 3=Kitchen, 4=Office
PRODUCTS = {
    1:  ("Studio Wireless Headphones", 1, 89.99, 4.6, True,
         {"audio": 0.9, "wearable": 0.3, "office": 0.6, "wireless": 0.7},
         "Closed-back wireless headphones tuned for focused studio listening at a desk."),
    2:  ("Reference Studio Monitors", 1, 149.00, 4.8, True,
         {"audio": 0.95, "wearable": 0.1, "office": 0.6, "wireless": 0.5},
         "Over-ear studio monitors with a flat response and Bluetooth for critical desk mixing."),
    3:  ("Sport Wireless Earbuds", 1, 59.99, 4.3, True,
         {"audio": 0.7, "wearable": 0.8, "outdoor": 0.1, "water": 0.5, "fitness": 0.8, "wireless": 0.7},
         "Sweatproof wireless earbuds that stay put during runs and gym workouts."),
    4:  ("Noise-Cancelling Earbuds", 1, 129.00, 4.5, True,
         {"audio": 0.85, "wearable": 0.7, "office": 0.2, "fitness": 0.1, "wireless": 0.8},
         "Active noise-cancelling earbuds with a wireless charging case for commutes."),
    5:  ("Waterproof Shower Speaker", 1, 24.99, 4.1, True,
         {"audio": 0.7, "outdoor": 0.3, "water": 0.9, "wireless": 0.6},
         "Compact waterproof Bluetooth speaker with a suction mount for the shower."),
    6:  ("Trail Running Vest", 2, 74.99, 4.4, True,
         {"wearable": 0.7, "outdoor": 0.8, "water": 0.6, "fitness": 0.8},
         "Lightweight hydration vest for trail running with a weatherproof shell."),
    7:  ("Waterproof Hiking Jacket", 2, 159.00, 4.7, True,
         {"wearable": 0.6, "outdoor": 0.9, "water": 0.95, "fitness": 0.2},
         "Breathable waterproof shell jacket for hiking in heavy rain."),
    8:  ("Rugged Wireless Headphones", 2, 45.00, 3.9, True,
         {"audio": 0.5, "wearable": 0.4, "outdoor": 0.8, "water": 0.7, "fitness": 0.3, "wireless": 0.6},
         "Rugged wireless headphones with a weatherproof, sweat-resistant build for the outdoors."),
    9:  ("Insulated Water Bottle", 2, 19.99, 4.2, True,
         {"wearable": 0.4, "outdoor": 0.7, "water": 0.8, "kitchen": 0.3, "fitness": 0.5},
         "Vacuum-insulated stainless steel water bottle that keeps drinks cold on the trail."),
    10: ("Camping Lantern", 2, 34.99, 4.0, False,
         {"wearable": 0.1, "outdoor": 0.9, "water": 0.3, "office": 0.1, "wireless": 0.2},
         "Rechargeable LED camping lantern with a collapsible, weather-resistant body."),
    11: ("Chef's Knife", 3, 79.00, 4.9, True,
         {"kitchen": 0.95, "office": 0.1},
         "Forged high-carbon chef's knife balanced for everyday kitchen prep."),
    12: ("Espresso Machine", 3, 249.00, 4.6, True,
         {"water": 0.2, "kitchen": 0.9, "office": 0.1, "wireless": 0.1},
         "Compact espresso machine with a steam wand for lattes at home."),
    13: ("Cast-Iron Skillet", 3, 39.99, 4.7, True,
         {"outdoor": 0.2, "kitchen": 0.9},
         "Pre-seasoned cast-iron skillet that moves from stovetop to oven to campfire."),
    14: ("Standing Desk", 4, 329.00, 4.5, True,
         {"office": 0.95, "fitness": 0.2},
         "Electric height-adjustable standing desk for a healthier home office."),
    15: ("Ergonomic Office Chair", 4, 199.00, 4.4, True,
         {"wearable": 0.2, "office": 0.9, "fitness": 0.3},
         "Mesh-back ergonomic office chair with lumbar support for long work sessions."),
    16: ("Wireless Mechanical Keyboard", 4, 99.00, 4.5, True,
         {"wearable": 0.1, "office": 0.8, "wireless": 0.7},
         "Low-profile wireless mechanical keyboard for a tidy desk setup."),
}


def embed(weights):
    """A product's weight dict -> a dense vector over AXES, in axis order."""
    return [round(weights.get(axis, 0.0), 3) for axis in AXES]


def sql_literal(text):
    return "'" + text.replace("'", "''") + "'"


def main():
    print("INSERT INTO products (id, name, category_id, price, rating, in_stock, description, embedding) VALUES")
    rows = []
    for pid, (name, cat, price, rating, in_stock, weights, desc) in PRODUCTS.items():
        vec = "[" + ",".join(str(x) for x in embed(weights)) + "]"
        rows.append(
            f"  ({pid}, {sql_literal(name)}, {cat}, {price}, {rating}, "
            f"{str(in_stock).lower()}, {sql_literal(desc)}, {sql_literal(vec)})"
        )
    print(",\n".join(rows) + ";")


if __name__ == "__main__":
    main()
