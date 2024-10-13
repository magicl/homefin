# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd


def create_aggregates(table, pnl=True, _prefix='') -> list[pd.DataFrame]:
    """
    Returns aggregates for groupings in table
    :param pnl: True if PNL else BS. Adds special aggregates depending on mode
    """
    # Names of items in group are the values in index after prefix, and before the next '/'
    subGroups = table.index.str.removeprefix(_prefix).str.split('/').str[0].unique()  # Maintains order

    groupOutput: list[pd.DataFrame] = []
    for groupName in subGroups:
        groupPrefix = f'{_prefix}{groupName}'
        groupItems = table[table.index.str.startswith(groupPrefix)]

        groupOutput.append(groupItems.sum().to_frame(name=f'{groupPrefix}/*').T)

        subOutput = create_aggregates(
            groupItems[groupItems.index.str.startswith(f'{groupPrefix}/')],
            pnl,
            f'{groupPrefix}/',
        )

        if subOutput:
            groupOutput += subOutput

    if _prefix == '':
        # Top level. Add special aggregation depending on type. First combine for easy lookup
        groupOutput = [pd.concat(groupOutput)]

        # Total sum
        table_sum = table.sum()
        groupOutput.append(table_sum.to_frame(name='** Sum Total').T)

        if pnl:
            groupOutput.append(
                (table_sum - groupOutput[0].loc['Investments/*']).to_frame(name='** Sum Ex Investments').T
            )

    return groupOutput
