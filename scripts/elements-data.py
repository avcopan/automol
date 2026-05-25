#!/usr/bin/env python
"""Update element data file."""

import json
from pathlib import Path

from mendeleev import element as mendeleev_element
from mendeleev.db import get_engine
from sqlalchemy import MetaData, Table, func, select

engine = get_engine()
metadata = MetaData()

elements = Table("elements", metadata, autoload_with=engine)
isotopes = Table("isotopes", metadata, autoload_with=engine)

abundance_ranked_isotopes = select(
    isotopes.c.atomic_number,
    isotopes.c.mass_number,
    isotopes.c.mass,
    # Partition by atomic number and order by abundance descending
    func.row_number()
    .over(
        partition_by=isotopes.c.atomic_number,
        order_by=isotopes.c.abundance.desc(),
    )
    .label("rank"),
).subquery()

primary_isotopes = (
    select(
        abundance_ranked_isotopes.c.atomic_number,
        abundance_ranked_isotopes.c.mass_number,
        abundance_ranked_isotopes.c.mass,
    )
    .where(abundance_ranked_isotopes.c.rank == 1)
    .subquery()
)

stmt = select(
    elements.c.symbol,
    elements.c.covalent_radius_pyykko,
    elements.c.group_id,
    elements.c.period,
    primary_isotopes.c.atomic_number,
    primary_isotopes.c.mass_number,
    primary_isotopes.c.mass,
).join(primary_isotopes, elements.c.atomic_number == primary_isotopes.c.atomic_number)

with engine.connect() as conn:
    result = conn.execute(stmt)
    elements_data = [
        {
            "Z": row.atomic_number,
            "A": row.mass_number,
            "group": row.group_id,
            "period": row.period,
            "symbol": row.symbol,
            "mass": row.mass,
            "covalent_radius": row.covalent_radius_pyykko / 100,
            "nvalence": mendeleev_element(row.atomic_number).nvalence(),
        }
        for row in result
    ]

elements_data_file = Path("src/automol/element/elements-data.json")

with elements_data_file.open("w", encoding="utf-8") as f:
    json.dump(elements_data, f, indent=2)
