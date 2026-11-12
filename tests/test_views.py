"""views.build gives one result per SELECT: what an order or a customer gets must not depend on scan order."""
import duckdb

from datamate import views

SELECT = """SELECT order_id, DATE '2026-01-05' AS order_date, NULL AS order_ts, NULL AS product_code, NULL AS product_group, item AS product_name,
       1 AS qty, NULL AS unit, amount, NULL AS customer_id, customer AS customer_name, staff, NULL AS channel, NULL AS payment,
       false AS is_cancelled, false AS is_returned, 'item' AS line_kind FROM lines"""


def shop(rows):
    db = duckdb.connect(":memory:")
    db.execute("CREATE TABLE lines (order_id VARCHAR, item VARCHAR, amount DOUBLE, customer VARCHAR, staff VARCHAR)")
    db.executemany("INSERT INTO lines VALUES (?, ?, ?, ?, ?)", rows)
    return db


def test_an_order_naming_two_customers_takes_the_one_on_most_lines():
    db = shop([("A", "x", 10, "Chị Lan", "Tí"), ("A", "y", 20, "Anh Tú", "Tèo"), ("A", "z", 30, "Anh Tú", "Tèo"),
               ("B", "x", 10, "Chị Lan", "Tí")])
    views.build(db, SELECT)
    assert db.execute("SELECT order_id, customer_key, customer_name, staff FROM v_orders ORDER BY 1").fetchall() == [
        ("A", "anhtu", "Anh Tú", "Tèo"), ("B", "chilan", "Chị Lan", "Tí")]


def test_ties_go_to_the_smaller_value_on_every_build():
    db = shop([("A", "x", 10, "Chị Lan", "Tí"), ("A", "y", 20, "Anh Tú", "Tèo"), ("B", "x", 5, None, None), ("C", "x", 5, "Cô Ba", "Tí")])
    seen = set()
    for _ in range(5):
        views.build(db, SELECT)
        seen.add(tuple(db.execute("SELECT order_id, customer_key, customer_name, staff FROM v_orders_all ORDER BY 1").fetchall()))
    assert seen == {(("A", "anhtu", "Anh Tú", "Tèo"), ("B", None, None, None), ("C", "coba", "Cô Ba", "Tí"))}


def test_a_customer_gets_its_commonest_spelling_then_the_first_in_order():
    db = shop([("A", "x", 10, "chị lan", None), ("B", "x", 10, "Chị Lan", None), ("C", "x", 10, "Chị Lan", None),
               ("D", "x", 10, "anh tú", None), ("E", "x", 10, "Anh Tú", None)])
    views.build(db, SELECT)
    assert db.execute("SELECT DISTINCT customer_key, customer_name FROM v_sales ORDER BY 1").fetchall() == [("anhtu", "Anh Tú"), ("chilan", "Chị Lan")]
    assert db.execute("SELECT count(*), count(DISTINCT order_id) FROM v_sales_all").fetchone() == (5, 5)  # the name join adds no line


def test_ingest_reads_negative_shorthand_mixed_dates_and_keeps_addresses_as_text():
    import pandas as pd
    from datamate.ingest import prepare, to_number
    s = pd.Series(["-1tr2", "-2tr050", "1tr2", "-500k", "1,45tr"], dtype="string")
    assert list(to_number(s)) == [-1200000, -2050000, 1200000, -500000, 1450000]
    df = prepare(pd.DataFrame({"Han tra": ["45897", "46128", "01/06/2026", "45859"] * 5,
                               "Addr": [f"{i} Đường số {i}" for i in range(20)],
                               "SL": ["1 chai", "2", "3 gói", "1"] * 5}, dtype=object))
    assert str(df["Han tra"].dtype).startswith("datetime64") and df["Han tra"].iloc[2] == pd.Timestamp("2026-06-01")
    assert df["Han tra"].iloc[0] == pd.Timestamp("2025-08-28")  # serial 45658 = 2025-01-01
    assert df["Addr"].dtype == "string" and df["Addr"].iloc[3] == "3 Đường số 3"
    assert list(df["SL"].iloc[:4]) == [1, 2, 3, 1] and "SL (đơn vị ghi kèm)" in df.columns
