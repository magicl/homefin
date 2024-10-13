# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import numpy as np
import pandas as pd

from .base import FinProvider


class PayPalProvider(FinProvider):
    """
    Paypal notes:
    - Must look at 'Currency' field. For each international transaction, there are two additional transactions on the same day to perform
      the conversion
    """

    statementPath = 'paypal'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath)

        # Fix data
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df['_cents'] = (
            df['Amount'].astype(str).str.replace(',', '').astype(float) * 100
        )  # Initial astype(str) due to read-in values being mixed float/str
        df['_description'] = df['Type'] + (' - ' + df['Name']).where(~df['Name'].isna(), '')

        # Match up currency conversions. I.e. for a NOK transaction, find its corresponding USD conversion, then replace the NOK value with the USD
        conversions = df[(df['Type'] == 'General Currency Conversion') & (df['Currency'] == 'USD')]
        conversions = conversions.set_index(['Date', 'Time'])

        df = df.join(conversions, on=['Date', 'Time'], rsuffix='_conv')

        # pd.set_option('display.max_rows', None)
        # breakpoint() #Mappings are wrong(!)

        # Remove conversion rows. No longer needed
        df = df.drop(df[df['Type'] == 'General Currency Conversion'].index)

        df['_ref'] = np.where(
            ~df['_cents_conv'].isna() & (df['Currency'] != 'USD'), 'Converted from NOK: ' + df['Amount'].astype(str), ''
        )
        df['_cents'] = df['_cents'].where(df['_cents_conv'].isna() | (df['Currency'] == 'USD'), df['_cents_conv'])

        return df[['_account', 'Date', '_cents', '_description', '_ref']]
