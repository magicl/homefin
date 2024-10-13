# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd

from .base import FinProvider


class TruistProvider(FinProvider):
    """
    Fidelity notes:
    - Fidelity detects and blocks selenium access, i.e. login in a selenium session does not work
    """

    statementPath = 'truist'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath)

        df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='%m/%d/%Y')
        df['Amount'] = df['Amount'].str.replace('$', '')
        df['_cents'] = (
            df['Amount'].where(~df['Amount'].str.startswith('('), '-' + df['Amount'].str.slice(1, -1)).astype(float)
            * 100
        )
        df['Check/Serial #'] = df['Check/Serial #'].fillna('')

        return df[['_account', 'Transaction Date', '_cents', 'Description', 'Check/Serial #']]

    async def fetchNewTransactions(self, page):
        await page.goto('https://www.truist.com')
