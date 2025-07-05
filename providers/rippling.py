# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import pandas as pd

from .base import FinProvider


class RipplingProvider(FinProvider):
    """
    Rippling Notes:
    - We are using a custom report in rippling. Each line represents a payrun, including salary,
      tax, 401k deductions, etc. The output of this provider breaks each of those into separate lines
    """

    statementPath = 'paychecks/rippling'

    def readTransactions(self):
        df = self.readDirectoryCSVs(self.statementPath)

        df['_date'] = pd.to_datetime(df['Pay run check date'], format='%Y-%m-%d')
        df = df[df['Pay run status'] == 'Paid']

        def select(col, desc, negate):
            sub = df.copy()
            sub = sub[~sub[col].isna()]

            sub['_cents'] = sub[col].astype(float) * 100
            sub = sub[sub['_cents'] > 0]
            sub['_description'] = sub['Employee'] + f' - {desc}'

            if negate:
                sub['_cents'] = -sub['_cents']

            return sub

        out = pd.concat(
            [
                select('Employee gross pay (USD)', 'gross pay', False),
                select('Employee net pay (USD)', 'net pay', True),
                select('Employee taxes (USD)', 'taxes', True),
                select('HSA (Employee, Flat amount) (USD)', 'HSA', True),
                select('Blue Cross (Employee, Flat amount) (USD)', 'health care', True),
                select('Roth 401K (Employee, Flat amount) (USD)', 'roth 401k', True),
                select('401K (Employee, Flat amount) (USD)', '401k', True),
                select('401K (Company, Flat amount) (USD)', 'contrib income', False),
                select('401K (Company, Flat amount) (USD)', '401k company contrib ', True),
                # Reimbursements are not included in gross pay. Create two entries, to transfer from
                # payroll payment and into the right account
                select('Reimbursements (USD)', 'reimbursements debit', True),
                select('Reimbursements (USD)', 'reimbursements credit', False),
            ]
        )
        out['Pay period'] = out['Pay period'].fillna('')

        return out[['_account', '_date', '_cents', '_description', 'Pay period']]
