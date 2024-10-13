# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd

from .base import FinProvider


class SchwabProvider(FinProvider):
    statementPath = 'schwab'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath)

        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

        deposits = (
            df['Deposit'].where(~df['Deposit'].isna(), '0').str.replace('$', '').str.replace(',', '').astype(float)
        )
        withdrawals = (
            df['Withdrawal']
            .where(~df['Withdrawal'].isna(), '0')
            .str.replace('$', '')
            .str.replace(',', '')
            .astype(float)
        )

        df['_cents'] = (deposits - withdrawals) * 100
        df['_ref'] = df['CheckNumber'].where(~df['CheckNumber'].isna(), '')

        return df[['_account', 'Date', '_cents', 'Description', '_ref']]
