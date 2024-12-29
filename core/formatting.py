# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd
from gspread_formatting import border  # pylint: disable=no-name-in-module
from gspread_formatting import borders  # pylint: disable=no-name-in-module
from gspread_formatting import cellFormat  # pylint: disable=no-name-in-module
from gspread_formatting import color  # pylint: disable=no-name-in-module
from gspread_formatting import numberFormat  # pylint: disable=no-name-in-module
from gspread_formatting import textFormat  # pylint: disable=no-name-in-module

######################################
# Output Styles
#####################################
# Date header
H = cellFormat(
    textFormat=textFormat(bold=True, foregroundColor=color(1, 1, 1)),
    backgroundColor=color(0, 0, 0),
    horizontalAlignment='RIGHT',
)

# Category column by level
# backgroundColor=color(1, 1, 1), foregroundColor=color(0, 0, 0)
C = [
    cellFormat(horizontalAlignment='LEFT', textFormat=textFormat(bold=True)),
    cellFormat(
        horizontalAlignment='LEFT', backgroundColor=color(1, 1, 1)
    ),  # No style. Bg color is set as empty is not allowed
    cellFormat(horizontalAlignment='LEFT', textFormat=textFormat(foregroundColor=color(0, 0, 1), italic=True)),
    cellFormat(horizontalAlignment='LEFT', textFormat=textFormat(foregroundColor=color(0, 0.7, 0.7), italic=True)),
]

# Value column by level
V = [
    cellFormat(textFormat=textFormat(foregroundColor=color(0, 0, 0), bold=True)),
    cellFormat(backgroundColor=color(1, 1, 1)),  # No style. Bg color is set as empty is not allowed
    cellFormat(textFormat=textFormat(foregroundColor=color(0, 0, 1), italic=True)),
    cellFormat(textFormat=textFormat(foregroundColor=color(0, 0.7, 0.7), italic=True)),
]

# Sum cells lines
# S = cellFormat(borders=borders(top=border(style='SOLID')), backgroundColor=color(0, 1, 0))
T = cellFormat(borders=borders(top=border(style='SOLID'), bottom=border(style='DOUBLE')))
# Dollar values
D = cellFormat(
    numberFormat=numberFormat(type='CURRENCY', pattern='[<0]-$#,##0;[>0]$#,##0;-'), horizontalAlignment='RIGHT'
)
P = cellFormat(numberFormat=numberFormat(type='PERCENT', pattern='0.00%'), horizontalAlignment='RIGHT')


def fmtDfToGridHierarchical(df: pd.DataFrame, dateFmt: str):
    """
    Turns dataframe into a list(list(...)) where outer list is row, and inner is column values
    Hierarchy is applied based on index, which is expected to be of format A/B/.../F
    """

    ######################################
    # Output
    #####################################
    out: list[list[tuple[Any, Any] | tuple[Any, Any, Any]]] = []
    groups = []

    def outputGroups(items, position, level=1, prefix=''):
        # Names of items in group are the values in index after prefix, and before the next '/'
        subGroups = items.index.str.removeprefix(prefix).str.split('/').str[0].unique()  # Maintains order

        groupOutput: list[list[tuple]] = []
        groupGroups = []
        for groupName in subGroups:
            if '*' in groupName:
                continue

            groupPrefix = f'{prefix}{groupName}'
            groupItems = items[items.index.str.startswith(groupPrefix)]
            positionStart = position + len(groupOutput)

            groupAgg = items.loc[f'{groupPrefix}/*'] if groupItems.shape[0] > 1 else groupItems.iloc[0]
            # groupAgg = groupItems.sum() if aggregator is None else aggregator(groupItems)
            visualPrefix = '      ' * (level - 2) + ' └─ ' if level > 1 else ''
            groupOutput.append(
                [(visualPrefix + groupName, C[level - 1]), *[(v, V[level - 1]) for v in groupAgg.to_list()]]
            )

            subOutput, subGroups = outputGroups(
                groupItems[groupItems.index.str.startswith(f'{groupPrefix}/')],
                position + len(groupOutput),
                level + 1,
                f'{groupPrefix}/',
            )

            if subOutput:
                groupOutput += subOutput
                groupGroups += [
                    {'start': positionStart, 'end': position + len(groupOutput) - 1, 'depth': level},
                    *subGroups,
                ]

        return groupOutput, groupGroups

    # Heading
    out.append([('', H), *[(c if isinstance(c, str) else c.strftime(dateFmt), H) for c in df.columns]])

    # Position is 2 due to heading above
    groupOutput, groups = outputGroups(df, 2)
    out += groupOutput

    # Append total sum
    # totalSum = df.sum() if aggregator is None else aggregator(df)
    if '** Sum Total' in df.index:
        totalAgg = df.loc['** Sum Total']
        out.append([('Sum Total', C[0], T), *[(v, T) for v in totalAgg.to_list()]])

    if '** Sum Ex Investments' in df.index:
        totalAgg = df.loc['** Sum Ex Investments']
        out.append([('Sum Ex Investments', C[0], T), *[(v, T) for v in totalAgg.to_list()]])

    return out, groups


def dfMixColumns(table: pd.DataFrame, others: list[pd.DataFrame], otherNames: list[Callable], groupColOffset=0):
    """
    Where columns match, columns are mixed between tables
    """
    if len(others) != len(otherNames):
        raise Exception('Length of others and otherNames not matching')

    table = table.copy()
    table.columns = table.columns.astype('object')  # Allow mixing of dates and str

    # Keep track of start and end of each grouping
    cols = {c.date(): (i, i) for i, c in enumerate(table.columns)}  # timestamp itself does not generate a useful hash

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        for otherI, other in enumerate(others):
            for c in other.columns:
                date = c.date()

                if date in cols:
                    idxFirst, idx = cols[date]
                    table.insert(idx + 1, otherNames[otherI](c), other[c], allow_duplicates=True)

                    # Update column locations
                    cols[date] = (idxFirst, idx + 1)
                    cols = {
                        c: (first, last) if first <= idxFirst else (first + 1, last + 1)
                        for c, (first, last) in cols.items()
                    }

    # Fix fragmentation due to lots of 'insert' calls above
    table = table.copy()

    # Create groupings for columns that have received mixed in data
    groups = [
        {'start': start + groupColOffset + 1, 'end': end + groupColOffset + 1, 'depth': 2}
        for start, end in cols.values()
        if start != end
    ]

    # Apply year-grouping as well, in case there are multiple columns in one year.
    groupStart = 0
    dateCount = 0
    prevYear = 0
    ci = 0
    for ci, c in enumerate(table.columns):
        if isinstance(c, pd.Timestamp):
            if c.year != prevYear:
                if ci - groupStart > 1 and dateCount > 1:
                    groups.append({'start': groupStart + groupColOffset + 1, 'end': ci + groupColOffset, 'depth': 1})

                prevYear = c.year
                groupStart = ci
                dateCount = 1
                continue

            dateCount += 1

    if ci - groupStart > 1 and dateCount > 1:
        # +1 in end because in loop above, we are on index after the one we are going for, but here we are on the last index, not the one after
        groups.append({'start': groupStart + groupColOffset + 1, 'end': ci + groupColOffset + 1, 'depth': 1})

    return table, groups


def fmtGridsAssemble(grids: list[list[list[Any]]]):
    """
    Creates a full grid of data based on input. Chunks is a 2D array of grids.
    Restrictions:
    - Grids are expected to match eachother. I.e. if two are next to eachother, their number of rows are expected to match.
      if they are on top of eachother, their number of columns is expected to match
    """

    out: list[list[Any]] = []
    for gridRow in grids:
        row = None
        for grid in gridRow:
            if row is None:
                # Shallow 2d copy. No need to re-create inidividual values
                row = [list(r) for r in grid]

            else:
                if len(grid) != len(row):
                    raise Exception('Height of grids next to eachother must match')

                # Add grid to current row
                for r, gr in zip(row, grid):
                    r += gr  # pylint: disable=redefined-loop-name

        if row is None:
            continue

        # Assemble rows
        if out is None:
            out = row

        else:
            out += row

    return out


def fmtGridApplyFmtFuncs(grids, funcs):
    for grid in grids:
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                for f in funcs:
                    res = f(grid, i, j)
                    if res is not None:
                        grid[i][j] = (*cell, *res)


def fmtBgGoodBad(value):
    """Given a value [1, -1], return gradually more red for negative, and gradually more green for positive values"""
    value = min(1, max(-1, value))
    scale = 1 - (abs(value) * 0.5)  # Transform abs(value) e [1, 0] -> [0.5, 1] for a softer max red/green
    return (
        cellFormat(backgroundColor=color(1, scale, scale))
        if value < 0
        else cellFormat(backgroundColor=color(scale, 1, scale))
    )
