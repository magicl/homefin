#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0 (the "License");
# Copyright 2024 Øivind Loe
# See LICENSE file or http://www.apache.org/licenses/LICENSE-2.0 for details.
# ~

import asyncio

import click
import environ
from playwright.async_api import async_playwright

from core.finance import Finance
from olib.py.utils.secrets import readFileSecret
from providers.base import Provider


@click.group()
@click.pass_context
def cli(ctx):
    env = environ.FileAwareEnv()
    environ.Env.read_env('.env', overwrite=True)
    environ.Env.read_env('.env.local', overwrite=True)

    ctx.obj = env


@cli.command(help='Load new data')
@click.option('--provider-name', default=None)
@click.option('--serial', default=False, is_flag=True)
@click.pass_context
def load(ctx, provider_name, serial, single_thread):
    providers = [
        p
        for p in Provider.getProviders().values()
        if provider_name is None or getattr(p, 'statementPath', None) == provider_name
    ]

    async def run_():
        async with async_playwright() as p:
            # Use persistent context so cookies can be stored, and we can usually avoid 2FA etc
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=ctx.obj.str('CHROME_USER_DATA_DIR'),
                executable_path=ctx.obj.str('CHROME_BINARY'),
                headless=False,
            )

            jobs = []
            for provider in providers:
                if hasattr(provider, 'fetchNewTransactions'):
                    page = await browser.new_page()
                    job = provider(None).fetchNewTransactions(page)
                    if serial:
                        await job
                    else:
                        jobs.append(job)

            if jobs:
                await asyncio.gather(*jobs)

    asyncio.run(run_())

    # with async_playwright() as p:
    #     #browser = p.chromium.launch(headless=False)

    #     page = browser.new_page()
    #     page.goto("https://truist.com")
    #     page.get_by_role('link', name='Sign In').click()
    #     breakpoint()
    #     print('next')


@cli.command(help='Update finances based on transaction exports')
@click.option('--serial', default=False, is_flag=True)
@click.pass_context
def run(ctx, serial):

    fin = Finance(
        ctx.obj.str('RECORDS_ROOT'),
        ctx.obj.str('GOOGLE_SHEET_ID'),
        readFileSecret(ctx.obj.str('GOOGLE_CREDS_FILE')),
        ctx.obj,
    )

    print('loading...')
    fin.load()

    print('connecting...')
    fin.connect(serial)

    print('updating...')
    fin.update()

    print('rendering...')
    fin.render()


if __name__ == '__main__':
    cli()  # pylint: disable=no-value-for-parameter
