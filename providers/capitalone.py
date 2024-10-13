# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~


import numpy as np
import pandas as pd

from .base import FinProvider


class CapitalOneProvider(FinProvider):
    statementPath = 'capitalone'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath)

        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='%m/%d/%y')
        df['_cents'] = df['Transaction Amount'].astype(float) * 100

        # For some transactions, the amount does not have the correct sign
        absCents = df['_cents'].abs()
        df['_cents'] = np.where(df['Transaction Type'] == 'Debit', -absCents, absCents)

        return df[['_account', 'Transaction Date', '_cents', 'Transaction Description', 'Transaction Type']]

    async def fetchNewTransactions(self, page):
        await page.goto('https://www.capitalone.com')
