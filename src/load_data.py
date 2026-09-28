"""Load the five source CSVs into pandas DataFrames with correct dtypes."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

DATE_COLS_TICKETS = ["created_at", "first_response_at", "resolved_at"]


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}. "
            f"Make sure the data/ directory contains the Vireo export files."
        )


def load_tickets(path: Path = config.TICKETS_CSV) -> pd.DataFrame:
    _require(path)
    df = pd.read_csv(path, dtype={"ticket_id": str, "order_id": str, "customer_id": str,
                                   "agent_id": str, "product_sku": str})
    for col in DATE_COLS_TICKETS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    logger.info("Loaded %d tickets from %s", len(df), path)
    return df


def load_agents(path: Path = config.AGENTS_CSV) -> pd.DataFrame:
    _require(path)
    df = pd.read_csv(path, dtype={"agent_id": str})
    df["from_date"] = pd.to_datetime(df["from_date"], errors="coerce")
    df["to_date"] = pd.to_datetime(df["to_date"], errors="coerce")
    logger.info("Loaded %d agent-roster rows from %s", len(df), path)
    return df


def load_customers(path: Path = config.CUSTOMERS_CSV) -> pd.DataFrame:
    _require(path)
    df = pd.read_csv(path, dtype={"customer_id": str})
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    logger.info("Loaded %d customers from %s", len(df), path)
    return df


def load_orders(path: Path = config.ORDERS_CSV) -> pd.DataFrame:
    _require(path)
    df = pd.read_csv(path, dtype={"order_id": str, "customer_id": str, "sku": str})
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    logger.info("Loaded %d orders from %s", len(df), path)
    return df


def load_products(path: Path = config.PRODUCTS_CSV) -> pd.DataFrame:
    _require(path)
    df = pd.read_csv(path, dtype={"sku": str})
    logger.info("Loaded %d products from %s", len(df), path)
    return df


def load_all(data_dir: Path = config.DATA_DIR) -> dict[str, pd.DataFrame]:
    return {
        "tickets": load_tickets(data_dir / "tickets.csv"),
        "agents": load_agents(data_dir / "agents.csv"),
        "customers": load_customers(data_dir / "customers.csv"),
        "orders": load_orders(data_dir / "orders.csv"),
        "products": load_products(data_dir / "products.csv"),
    }
