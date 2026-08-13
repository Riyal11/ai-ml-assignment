"""Parse SuperStore invoice PDF text into schema-aligned invoice records."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from dateutil import parser as date_parser

logger = logging.getLogger(__name__)

VENDOR_NAME = "SuperStore"
CURRENCY = "USD"
MONEY_TOLERANCE = Decimal("0.01")

_INVOICE_NUMBER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"INVOICE\s*#\s*(\d+)", re.IGNORECASE),
    re.compile(r"Row\s*ID[:\s]+(\d+)", re.IGNORECASE),
    re.compile(r"#\s*(\d+)\b"),
)
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Order\s*Date[:\s]+([A-Za-z]+\s*\d{1,2}\s*\d{4})", re.IGNORECASE),
    re.compile(
        r"Date[:\s]+([A-Za-z]{3}\d{6}|[A-Za-z]{3}\s+\d{1,2}\s+\d{4})",
        re.IGNORECASE,
    ),
)
_BILL_TO_PATTERN = re.compile(
    r"Bill\s*To\s*:\s*(.+?)(?=\s*Ship\s*To\s*:|\s*(?:Item|Product)\b|\n\n)",
    re.IGNORECASE | re.DOTALL,
)
_SUBTOTAL_PATTERN = re.compile(r"Subtotal\s*:\s*\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE)
_DISCOUNT_PATTERN = re.compile(
    r"Discount\s*\(\s*(\d+(?:\.\d+)?)\s*%\s*\)\s*:\s*\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_SHIPPING_PATTERN = re.compile(r"Shipping\s*:\s*\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE)
_TOTAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Total\s*Amount\s*Payable\s*:\s*\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"Balance\s*Due\s*:\s*\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"(?<!Sub)Total\s*:\s*\$?([\d,]+(?:\.\d+)?)", re.IGNORECASE),
)
_ORDER_ID_PATTERN = re.compile(r"Order\s*ID\s*:\s*(\S+)", re.IGNORECASE)
_COMPACT_DATE_PATTERN = re.compile(r"^([A-Za-z]{3})(\d{2})(\d{4})$")

_DESCRIPTION_HEADERS = ("product name", "item", "description", "item/product name")
_QUANTITY_HEADERS = ("quantity", "qty")
_UNIT_PRICE_HEADERS = ("rate", "unit cost", "unit price", "rate/unit cost")
_LINE_ITEM_TRAILING = re.compile(
    r"^(?P<description>.+?)\s+"
    r"(?P<quantity>\d+)\s+"
    r"(?P<unit_price>\$?[\d,]+(?:\.\d+)?)\s+"
    r"(?P<amount>\$?[\d,]+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class DiscountInfo:
    """Parsed discount line from a SuperStore invoice."""

    percent: Decimal
    amount: Decimal


@dataclass
class ParsedSuperstoreInvoice:
    """Structured extraction result before schema mapping."""

    invoice_number: str
    vendor_name: str = VENDOR_NAME
    invoice_date: str = ""
    customer_name: str = ""
    line_items: list[dict[str, Any]] = field(default_factory=list)
    subtotal: Decimal = Decimal("0")
    discount: DiscountInfo | None = None
    shipping: Decimal = Decimal("0")
    total_amount: Decimal = Decimal("0")
    order_id: str | None = None
    currency: str = CURRENCY
    total_validation_warning: str | None = None


def parse_money(value: str) -> Decimal:
    """Parse a currency string like ``$1,880.45`` into a ``Decimal``."""
    cleaned = value.strip().replace("$", "").replace(",", "")
    if not cleaned:
        raise ValueError("empty money value")
    try:
        return Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"invalid money value: {value!r}") from exc


def _parse_quantity(value: str) -> int:
    """Parse a quantity column value into an integer."""
    cleaned = value.strip().replace(",", "")
    if not cleaned.isdigit():
        raise ValueError(f"invalid quantity value: {value!r}")
    return int(cleaned)


def _parse_line_item_row(
    description: str,
    quantity_raw: str,
    unit_price_raw: str,
) -> dict[str, Any] | None:
    """Parse one line-item row, returning ``None`` when the row is invalid."""
    description = description.strip()
    if not description:
        return None
    if description.lower() in {"subtotal", "total", "shipping", "discount"}:
        return None
    try:
        quantity = _parse_quantity(quantity_raw)
        unit_price = parse_money(unit_price_raw)
    except ValueError:
        return None
    return {
        "description": description,
        "quantity": quantity,
        "unit_price": float(unit_price),
    }


def normalize_invoice_date(raw: str) -> str:
    """Normalize varied SuperStore date strings to ``YYYY-MM-DD``."""
    text = re.sub(r"\s+", " ", raw.strip())
    compact = text.replace(" ", "")
    compact_match = _COMPACT_DATE_PATTERN.match(compact)
    if compact_match:
        month, day, year = compact_match.groups()
        text = f"{month} {int(day)} {year}"
    parsed = date_parser.parse(text, dayfirst=False)
    return parsed.date().isoformat()


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return str(match.group(1).strip())
    return None


def _normalize_header(value: str | None) -> str:
    return (value or "").strip().lower()


def _column_index(headers: list[str | None], aliases: tuple[str, ...]) -> int | None:
    normalized = [_normalize_header(header) for header in headers]
    for idx, header in enumerate(normalized):
        if any(alias in header for alias in aliases):
            return idx
    return None


def parse_line_items_from_table(table: list[list[str | None]]) -> list[dict[str, Any]]:
    """Parse line items from a pdfplumber-style table."""
    if not table:
        return []

    header_row: list[str | None] | None = None
    header_index = 0
    for idx, row in enumerate(table):
        normalized = " ".join(_normalize_header(cell) for cell in row if cell)
        if any(alias in normalized for alias in _DESCRIPTION_HEADERS) and any(
            alias in normalized for alias in _QUANTITY_HEADERS
        ):
            header_row = row
            header_index = idx
            break

    if header_row is None:
        return []

    desc_idx = _column_index(header_row, _DESCRIPTION_HEADERS)
    qty_idx = _column_index(header_row, _QUANTITY_HEADERS)
    price_idx = _column_index(header_row, _UNIT_PRICE_HEADERS)
    if desc_idx is None or qty_idx is None or price_idx is None:
        return []

    items: list[dict[str, Any]] = []
    for row in table[header_index + 1 :]:
        if not row or all(not (cell or "").strip() for cell in row):
            continue
        description = (row[desc_idx] or "").strip()
        quantity_raw = (row[qty_idx] or "").strip()
        unit_price_raw = (row[price_idx] or "").strip()
        if not description or not quantity_raw or not unit_price_raw:
            continue
        item = _parse_line_item_row(description, quantity_raw, unit_price_raw)
        if item is not None:
            items.append(item)
    return items


def parse_line_items_from_text(text: str) -> list[dict[str, Any]]:
    """Fallback line-item parser when table extraction is unavailable."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_idx: int | None = None
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(alias in lowered for alias in _DESCRIPTION_HEADERS) and any(
            alias in lowered for alias in _QUANTITY_HEADERS
        ):
            header_idx = idx
            break
    if header_idx is None:
        return []

    items: list[dict[str, Any]] = []
    for line in lines[header_idx + 1 :]:
        lowered = line.lower()
        if lowered.startswith(("subtotal", "discount", "shipping", "total", "balance due")):
            break

        description: str | None = None
        quantity_raw: str | None = None
        unit_price_raw: str | None = None

        parts = re.split(r"\s{2,}|\t+", line)
        if len(parts) >= 4:
            description = " ".join(part.strip() for part in parts[:-3])
            quantity_raw = parts[-3].strip()
            unit_price_raw = parts[-2].strip()
        else:
            match = _LINE_ITEM_TRAILING.match(line)
            if match:
                description = match.group("description")
                quantity_raw = match.group("quantity")
                unit_price_raw = match.group("unit_price")

        if description is None or quantity_raw is None or unit_price_raw is None:
            continue

        item = _parse_line_item_row(description, quantity_raw, unit_price_raw)
        if item is not None:
            items.append(item)
    return items


def validate_total_amount(
    subtotal: Decimal,
    discount: DiscountInfo | None,
    shipping: Decimal,
    total_amount: Decimal,
) -> str | None:
    """Return a warning message when totals do not reconcile."""
    discount_amount = discount.amount if discount else Decimal("0")
    computed = subtotal - discount_amount + shipping
    if abs(computed - total_amount) > MONEY_TOLERANCE:
        return (
            f"computed total {computed} != PDF total {total_amount} "
            f"(subtotal={subtotal}, discount={discount_amount}, shipping={shipping})"
        )
    return None


def to_schema_target(parsed: ParsedSuperstoreInvoice) -> dict[str, Any]:
    """Map a parsed invoice to the strict invoice schema target dict."""
    return {
        "invoice_number": parsed.invoice_number,
        "vendor_name": parsed.vendor_name,
        "invoice_date": parsed.invoice_date,
        "line_items": parsed.line_items,
        "subtotal": float(parsed.subtotal),
        "tax_amount": 0.0,
        "total_amount": float(parsed.total_amount),
        "currency": parsed.currency,
    }


def parse_superstore_invoice_text(
    text: str,
    tables: list[list[list[str | None]]] | None = None,
) -> ParsedSuperstoreInvoice:
    """Parse extracted PDF text (and optional tables) into invoice fields."""
    invoice_number = _first_match(_INVOICE_NUMBER_PATTERNS, text)
    if not invoice_number:
        raise ValueError("invoice_number not found")

    date_raw = _first_match(_DATE_PATTERNS, text)
    if not date_raw:
        raise ValueError("invoice_date not found")
    invoice_date = normalize_invoice_date(date_raw)

    bill_to_match = _BILL_TO_PATTERN.search(text)
    customer_name = bill_to_match.group(1).strip() if bill_to_match else ""

    subtotal_match = _SUBTOTAL_PATTERN.search(text)
    if not subtotal_match:
        raise ValueError("subtotal not found")
    subtotal = parse_money(subtotal_match.group(1))

    discount: DiscountInfo | None = None
    discount_match = _DISCOUNT_PATTERN.search(text)
    if discount_match:
        discount = DiscountInfo(
            percent=Decimal(discount_match.group(1)),
            amount=parse_money(discount_match.group(2)),
        )

    shipping_match = _SHIPPING_PATTERN.search(text)
    shipping = parse_money(shipping_match.group(1)) if shipping_match else Decimal("0")

    total_raw = _first_match(_TOTAL_PATTERNS, text)
    if not total_raw:
        raise ValueError("total_amount not found")
    total_amount = parse_money(total_raw)

    order_match = _ORDER_ID_PATTERN.search(text)
    order_id = order_match.group(1).strip() if order_match else None
    if order_id:
        logger.info("Parsed order_id=%s for invoice %s", order_id, invoice_number)

    line_items: list[dict[str, Any]] = []
    if tables:
        for table in tables:
            parsed_items = parse_line_items_from_table(table)
            if parsed_items:
                line_items.extend(parsed_items)
    if not line_items:
        line_items = parse_line_items_from_text(text)
    if not line_items:
        raise ValueError("line_items not found")

    warning = validate_total_amount(subtotal, discount, shipping, total_amount)
    if warning:
        logger.warning("Invoice %s total mismatch: %s", invoice_number, warning)

    return ParsedSuperstoreInvoice(
        invoice_number=invoice_number,
        invoice_date=invoice_date,
        customer_name=customer_name,
        line_items=line_items,
        subtotal=subtotal,
        discount=discount,
        shipping=shipping,
        total_amount=total_amount,
        order_id=order_id,
        total_validation_warning=warning,
    )
