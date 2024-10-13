# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd

from .base import FinProvider


class VenmoProvider(FinProvider):
    statementPath = 'venmo'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath, skipRows=2)

        df = df[~df['ID'].isna()]

        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%dT', exact=False)
        df['_cents'] = df['Amount (total)'].str.replace(' ', '').str.replace('$', '').astype(float) * 100
        df['_description'] = (
            'From '
            + df['From'].fillna('?')
            + ' to '
            + df['To'].fillna(df['Destination'].fillna('?'))
            + ' for '
            + df['Note'].fillna('')
        )

        df['Funding Source'] = df['Funding Source'].fillna('')

        # Add in payment transactions, since they are not part of venmo's dataset. Only do this when "Destination" has not been set.
        payments = df[df['Destination'].isna()].copy()
        payments['_description'] = 'Assumed Payment from ' + payments['Funding Source'].where(
            ~payments['Funding Source'].isna(), '??'
        )
        payments['_cents'] = -payments['_cents']

        df = pd.concat([df, payments])
        df['memo'] = df['Type'] + ' / ' + df['Funding Source']

        return df[['_account', 'Datetime', '_cents', '_description', 'memo']]
