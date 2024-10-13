# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import re

import numpy as np
import pandas as pd

from olib.py.utils.clients.google.sheets import gsReadDataFrame

from .utils import create_aggregates


class Budget:
    def __init__(self, finance):
        self.fin = finance
        self.yearBudget = None
        self.monthBudget = None

    def load(self):
        """
        Read and process all budgets. They have sheet name Budget-YYY
        """

        # pnlYearly = self.fin.dataPNL('YE', add_aggregates=False) * 100
        # pnlMonthly = self.fin.dataPNL('ME', add_aggregates=False) * 100
        pnlYearly = self.fin.dsY.pnl[~self.fin.dsY.pnl.index.str.contains('*', regex=False)] * 100
        # pnlMonthly = self.fin.dsM.pnl[~self.fin.dsM.pnl.index.str.contains('*', regex=False)] * 100

        # now = pd.Timestamp(datetime.date.today())

        yearBudgetFrames = []
        monthBudgetFrames = []

        # Work on one year at a time, as spreading rules apply to individual years.. Could have done it all at once, but no need
        for sheet in self.fin.gs.worksheets():
            m = re.match(r'Budget-(\d{4})', sheet.title)
            if m is None:
                continue

            year = int(m.group(1))
            print(f'  Budget: {year}')

            # Budget input
            budgetData = gsReadDataFrame(sheet, gs=self.fin.gs, skipRows=3)
            budgetData = budgetData[~budgetData['Categories'].isna()]
            budgetData.set_index('Categories', inplace=True)
            budgetData['Type'] = budgetData['Type'].str.lower()

            # Warn on missing categories in budget. In order for budget to stay ok, new categories must be back-added in
            for missingCat in set(pnlYearly.index) - set(budgetData.index):
                print(f'    missing budget category: {missingCat}')

            # Warn on unknown categories in budget
            for unknownCat in set(budgetData.index) - set(pnlYearly.index):
                print(f'    unknown budget category: {unknownCat}')

            budgetData = budgetData.reindex(pnlYearly.index)  # Add in missing categories

            monthInput = budgetData[[c for c in budgetData.columns if c.endswith(f'/{year}')]] * 100
            monthInput.columns = pd.to_datetime(monthInput.columns)

            yearInput = budgetData['Whole Year'] * 100
            yearInput.name = monthInput.columns[-1]

            # Fill in yearly and monthly budget based on last years spend where it has not been specified
            # monthInputGiven = monthInput.apply(lambda row: row.notna().any(), axis=1)
            monthInputGiven = ~np.isnan(monthInput.values).all(axis=1)

            yearBudget = np.where(
                ~yearInput.isna(),
                yearInput,
                np.where(
                    monthInputGiven,
                    monthInput.fillna(0).apply(lambda row: row.sum(), axis=1),  # row-wise sum
                    pnlYearly[yearInput.name.replace(year=yearInput.name.year - 1)],
                ),
            )

            monthBudget = np.where(
                np.repeat(monthInputGiven[:, np.newaxis], 12, axis=1),
                monthInput.fillna(0),
                np.repeat(yearBudget[:, np.newaxis] / 12, 12, axis=1),
            )

            # Update future month budgets based on rules. Do this iteratively so that each month's budget
            # does not change as we add more months into the mix
            # monthsPassed = len([c for c in monthInput.columns if c < now])

            # if monthsPassed > 0:
            #     for month in range(monthsPassed):
            #         spent = pnlMonthly.values[:, month]
            #         budget = monthBudget[:, month]
            #         surplus = budget - spent  # Negative if we spent less than budget, positive if we spent more

            #         futureEdit = np.where(
            #             budgetData['Type'] == 'yearlong',
            #             surplus,  # For yearlong budget surplus can affect up or down
            #             np.where(
            #                 budget <= 0,
            #                 np.maximum(
            #                     surplus, 0
            #                 ),  # For monthly budget, only "too much spend" is propagated. Income is not propagated at all
            #                 0,
            #             ),
            #         )

            #         monthNumber = month + 1
            #         monthBudget[:, monthNumber:] += np.repeat(
            #             futureEdit[:, np.newaxis] / (12 - monthNumber), 12 - monthNumber, axis=1
            #         )

            yearBudgetFrames.append(
                pd.DataFrame(yearBudget[:, np.newaxis], index=monthInput.index, columns=[yearInput.name])
            )
            monthBudgetFrames.append(pd.DataFrame(monthBudget, index=monthInput.index, columns=monthInput.columns))

        self.yearBudget = pd.concat(yearBudgetFrames, axis=1)
        self.monthBudget = pd.concat(monthBudgetFrames, axis=1)

    def dataBudget(self, freq, dataPNL, dataBS):
        assert self.yearBudget is not None and self.monthBudget is not None  # nosec

        if freq == 'YE':
            budget = self.yearBudget / 100
        elif freq == 'ME':
            budget = self.monthBudget / 100
        else:
            raise Exception(f'unknown freq: {freq}')

        # Match up budget to PNL, i.e. find shared columns
        shared = list(set(dataPNL.columns) & set(budget.columns))
        missing = list(set(dataPNL.columns) - set(budget.columns))

        # Scale last column by how far through it we are if in Year mode (YE)
        # progress = (pd.Timestamp.now() - dataPNL.columns[-2]) / (dataPNL.columns[-1] - dataPNL.columns[-2])
        if freq == 'YE':
            moBudget = self.monthBudget / 100

            # Include budget for months that have passed completely
            budget.iloc[:, -1] = moBudget.loc[
                :, (moBudget.columns > dataPNL.columns[-2]) & (moBudget.columns < pd.Timestamp.now())
            ].sum(axis=1)

        pnl = dataPNL[shared]
        pnlBudget = budget[shared].copy()

        # Sort columns
        pnlBudget = pnlBudget.reindex(sorted(pnlBudget.columns), axis=1)

        # Add aggregates
        pnlBudget = pd.concat([pnlBudget, *create_aggregates(pnlBudget, pnl=True)])

        pnlDelta = pnl - pnlBudget
        pnlDelta = pnlDelta.reindex(sorted(pnlDelta.columns), axis=1)

        # Fill columns not in budget with empty data
        for c in missing:
            pnlBudget[c] = 0
            pnlDelta[c] = 0

        return pnlDelta, pnlBudget
