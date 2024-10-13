# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd

from .base import FinProvider


class FidelityProvider(FinProvider):
    """
    Fidelity notes:
    - Fidelity detects and blocks selenium access, i.e. login in a selenium session does not work
    """

    statementPath = 'fidelity/cc'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath)

        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df['_cents'] = df['Amount'].astype(float) * 100

        # Fidelity marks recent records with Memo containing 00000; but then in later statements,
        # they get a value instead of the 00000. This leads to duplication as they look different.
        # Filter out the records with 00000; in memo
        df = df[~df['Memo'].str.contains('; 00000;')]
        df['_account'] = 'fidelity/' + df['_account']

        return df[['_account', 'Date', '_cents', 'Name', 'Memo']]
