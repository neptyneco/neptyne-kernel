from kernel.dash import Dash as _Dash
from kernel.proxied_apis import start_api_proxying as _start_api_proxying
N_ = _Dash.instance()
_start_api_proxying()
if N_.in_gs_mode:
    from kernel.kernel_globals.gsheets import *
else:
    from kernel.kernel_globals.core import *

import neptyne as nt
import streamlit as st

import random
from datetime import date



@nt.streamlit(
    public=True,
)
def app():
    email = st.experimental_user["email"]
    st.title(f"Hello, {email}")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)
    st.write(N_[0, 0, 0])


def random_date():
    return date(
        random.randint(1900, 2100),
        random.randint(1, 12),
        random.randint(1, 28),
    )


def random_str():
    return ''.join([chr(ord("a") + random.randint(0, 25)) for _ in range(4)])


def random_row():
    return [
        random_str(),
        random_date(),
        random.randint(1, 100),
    ]


def do_test():
    test_insert_zero()
    test_insert_end()
    test_append()


def test_insert_zero(cr):
    cr.insert_row(0, random_row())


def test_insert_end(cr):
    cr.insert_row(7, random_row())


def test_append(cr):
    cr.append_row(random_row())


def test_rows():
    with nt.sheets["Rows"]:
        test_append(N_[0, 2, 0, -1, 0])
        test_insert_zero(N_[4, 6, 0, -1, 0])
        test_insert_end(N_[8, 10, 0, -1, 0])

if __name__ == '__main__':
    app()